/-- Given a function `f : E → F`, a formal multilinear series `p` and `n : ℕ`, we say that
`f` has `p` as a finite power series on the ball of radius `r > 0` around `x` if
`f (x + y) = ∑' pₘ yᵐ` for all `‖y‖ < r` and `pₙ = 0` for `n ≤ m`. -/
structure HasFiniteFPowerSeriesOnBall (f : E → F) (p : FormalMultilinearSeries 𝕜 E F) (x : E)
    (n : ℕ) (r : ℝ≥0∞) extends HasFPowerSeriesOnBall f p x r : Prop where
  finite : ∀ (m : ℕ), n ≤ m → p m = 0


theorem HasFiniteFPowerSeriesOnBall.mk' {f : E → F} {p : FormalMultilinearSeries 𝕜 E F} {x : E}
    {n : ℕ} {r : ℝ≥0∞} (finite : ∀ (m : ℕ), n ≤ m → p m = 0) (pos : 0 < r)
    (sum_eq : ∀ y ∈ EMetric.ball 0 r, (∑ i ∈ Finset.range n, p i fun _ ↦ y) = f (x + y)) :
    HasFiniteFPowerSeriesOnBall f p x n r where
  r_le := p.radius_eq_top_of_eventually_eq_zero (Filter.eventually_atTop.mpr ⟨n, finite⟩) ▸ le_top
  r_pos := pos
  hasSum hy := sum_eq _ hy ▸ hasSum_sum_of_ne_finset_zero fun m hm ↦ by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      n : Nat
      r : ENNReal
      finite : ∀ (m : Nat), LE.le n m → Eq (p m) 0
      pos : LT.lt 0 r
      sum_eq : ∀ (y : E), Membership.mem (EMetric.ball 0 r) y → Eq ((Finset.range n) …
      y✝ : E
      hy : Membership.mem (EMetric.ball 0 r) y✝
      m : Nat
      hm : Not (Membership.mem (Finset.range n) m)
      ⊢ Eq ((p m) fun x => y✝) 0
    -/
    rw [Finset.mem_range, not_lt] at hm; rw [finite m hm]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/
  finite := finite


/-- Given a function `f : E → F`, a formal multilinear series `p` and `n : ℕ`, we say that
`f` has `p` as a finite power series around `x` if `f (x + y) = ∑' pₙ yⁿ` for all `y` in a
neighborhood of `0`and `pₙ = 0` for `n ≤ m`. -/
def HasFiniteFPowerSeriesAt (f : E → F) (p : FormalMultilinearSeries 𝕜 E F) (x : E) (n : ℕ) :=
  ∃ r, HasFiniteFPowerSeriesOnBall f p x n r


theorem HasFiniteFPowerSeriesAt.toHasFPowerSeriesAt
    (hf : HasFiniteFPowerSeriesAt f p x n) : HasFPowerSeriesAt f p x :=
  let ⟨r, hf⟩ := hf
  ⟨r, hf.toHasFPowerSeriesOnBall⟩


theorem HasFiniteFPowerSeriesAt.finite (hf : HasFiniteFPowerSeriesAt f p x n) :
    ∀ m : ℕ, n ≤ m → p m = 0 := let ⟨_, hf⟩ := hf; hf.finite


/-- Given a function `f : E → F`, we say that `f` is continuously polynomial (cpolynomial)
at `x` if it admits a finite power series expansion around `x`. -/
def CPolynomialAt (f : E → F) (x : E) :=
  ∃ (p : FormalMultilinearSeries 𝕜 E F) (n : ℕ), HasFiniteFPowerSeriesAt f p x n


/-- Given a function `f : E → F`, we say that `f` is continuously polynomial on a set `s`
if it is continuously polynomial around every point of `s`. -/
def CPolynomialOn (f : E → F) (s : Set E) :=
  ∀ x, x ∈ s → CPolynomialAt 𝕜 f x


theorem HasFiniteFPowerSeriesOnBall.hasFiniteFPowerSeriesAt
    (hf : HasFiniteFPowerSeriesOnBall f p x n r) :
    HasFiniteFPowerSeriesAt f p x n :=
  ⟨r, hf⟩


theorem HasFiniteFPowerSeriesAt.cPolynomialAt (hf : HasFiniteFPowerSeriesAt f p x n) :
    CPolynomialAt 𝕜 f x :=
  ⟨p, n, hf⟩


theorem HasFiniteFPowerSeriesOnBall.cPolynomialAt (hf : HasFiniteFPowerSeriesOnBall f p x n r) :
    CPolynomialAt 𝕜 f x :=
  hf.hasFiniteFPowerSeriesAt.cPolynomialAt


theorem CPolynomialAt.analyticAt (hf : CPolynomialAt 𝕜 f x) : AnalyticAt 𝕜 f x :=
  let ⟨p, _, hp⟩ := hf
  ⟨p, hp.toHasFPowerSeriesAt⟩


theorem CPolynomialAt.analyticWithinAt {s : Set E} (hf : CPolynomialAt 𝕜 f x) :
    AnalyticWithinAt 𝕜 f s x :=
  hf.analyticAt.analyticWithinAt


theorem CPolynomialOn.analyticOnNhd {s : Set E} (hf : CPolynomialOn 𝕜 f s) : AnalyticOnNhd 𝕜 f s :=
  fun x hx ↦ (hf x hx).analyticAt


theorem CPolynomialOn.analyticOn {s : Set E} (hf : CPolynomialOn 𝕜 f s) : AnalyticOn 𝕜 f s :=
  hf.analyticOnNhd.analyticOn


theorem HasFiniteFPowerSeriesOnBall.congr (hf : HasFiniteFPowerSeriesOnBall f p x n r)
    (hg : EqOn f g (EMetric.ball x r)) : HasFiniteFPowerSeriesOnBall g p x n r :=
  ⟨hf.1.congr hg, hf.finite⟩


/-- If a function `f` has a finite power series `p` around `x`, then the function
`z ↦ f (z - y)` has the same finite power series around `x + y`. -/
theorem HasFiniteFPowerSeriesOnBall.comp_sub (hf : HasFiniteFPowerSeriesOnBall f p x n r) (y : E) :
    HasFiniteFPowerSeriesOnBall (fun z => f (z - y)) p (x + y) n r :=
  ⟨hf.1.comp_sub y, hf.finite⟩


theorem HasFiniteFPowerSeriesOnBall.mono (hf : HasFiniteFPowerSeriesOnBall f p x n r)
    (r'_pos : 0 < r') (hr : r' ≤ r) : HasFiniteFPowerSeriesOnBall f p x n r' :=
  ⟨hf.1.mono r'_pos hr, hf.finite⟩


theorem HasFiniteFPowerSeriesAt.congr (hf : HasFiniteFPowerSeriesAt f p x n) (hg : f =ᶠ[𝓝 x] g) :
    HasFiniteFPowerSeriesAt g p x n :=
  Exists.imp (fun _ hg ↦ ⟨hg, hf.finite⟩) (hf.toHasFPowerSeriesAt.congr hg)


protected theorem HasFiniteFPowerSeriesAt.eventually (hf : HasFiniteFPowerSeriesAt f p x n) :
    ∀ᶠ r : ℝ≥0∞ in 𝓝[>] 0, HasFiniteFPowerSeriesOnBall f p x n r :=
  hf.toHasFPowerSeriesAt.eventually.mono fun _ h ↦ ⟨h, hf.finite⟩


theorem hasFiniteFPowerSeriesOnBall_const {c : F} {e : E} :
    HasFiniteFPowerSeriesOnBall (fun _ => c) (constFormalMultilinearSeries 𝕜 E c) e 1 ⊤ :=
  ⟨hasFPowerSeriesOnBall_const, fun n hn ↦ constFormalMultilinearSeries_apply (id hn : 0 < n).ne'⟩


theorem hasFiniteFPowerSeriesAt_const {c : F} {e : E} :
    HasFiniteFPowerSeriesAt (fun _ => c) (constFormalMultilinearSeries 𝕜 E c) e 1 :=
  ⟨⊤, hasFiniteFPowerSeriesOnBall_const⟩


theorem CPolynomialAt_const {v : F} : CPolynomialAt 𝕜 (fun _ => v) x :=
  ⟨constFormalMultilinearSeries 𝕜 E v, 1, hasFiniteFPowerSeriesAt_const⟩


theorem CPolynomialOn_const {v : F} {s : Set E} : CPolynomialOn 𝕜 (fun _ => v) s :=
  fun _ _ => CPolynomialAt_const


theorem HasFiniteFPowerSeriesOnBall.add (hf : HasFiniteFPowerSeriesOnBall f pf x n r)
    (hg : HasFiniteFPowerSeriesOnBall g pg x m r) :
    HasFiniteFPowerSeriesOnBall (f + g) (pf + pg) x (max n m) r :=
  ⟨hf.1.add hg.1, fun N hN ↦ by
    rw [Pi.add_apply, hf.finite _ ((le_max_left n m).trans hN),
        hg.finite _ ((le_max_right n m).trans hN), zero_add]⟩


theorem HasFiniteFPowerSeriesAt.add (hf : HasFiniteFPowerSeriesAt f pf x n)
    (hg : HasFiniteFPowerSeriesAt g pg x m) :
    HasFiniteFPowerSeriesAt (f + g) (pf + pg) x (max n m) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    x : E
    n m : Nat
    hf : HasFiniteFPowerSeriesAt f pf x n
    hg : HasFiniteFPowerSeriesAt g pg x m
    ⊢ HasFiniteFPowerSeriesAt (HAdd.hAdd f g) (HAdd.hAdd pf pg) x (Max.max n m)
  -/
  rcases (hf.eventually.and hg.eventually).exists with ⟨r, hr⟩
  /-
    case intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    x : E
    n m : Nat
    hf : HasFiniteFPowerSeriesAt f pf x n
    hg : HasFiniteFPowerSeriesAt g pg x m
    r : ENNReal
    hr : And (HasFiniteFPowerSeriesOnBall f pf x n r) (HasFiniteFPowerSeriesOnBall …
    ⊢ HasFiniteFPowerSeriesAt (HAdd.hAdd f g) (HAdd.hAdd pf pg) x (Max.max n m)
  -/
  exact ⟨r, hr.1.add hr.2⟩
  /-
    🎉 no goals
  -/


theorem CPolynomialAt.congr (hf : CPolynomialAt 𝕜 f x) (hg : f =ᶠ[𝓝 x] g) : CPolynomialAt 𝕜 g x :=
  let ⟨_, _, hpf⟩ := hf
  (hpf.congr hg).cPolynomialAt


theorem CPolynomialAt_congr (h : f =ᶠ[𝓝 x] g) : CPolynomialAt 𝕜 f x ↔ CPolynomialAt 𝕜 g x :=
  ⟨fun hf ↦ hf.congr h, fun hg ↦ hg.congr h.symm⟩


theorem CPolynomialAt.add (hf : CPolynomialAt 𝕜 f x) (hg : CPolynomialAt 𝕜 g x) :
    CPolynomialAt 𝕜 (f + g) x :=
  let ⟨_, _, hpf⟩ := hf
  let ⟨_, _, hqf⟩ := hg
  (hpf.add hqf).cPolynomialAt


theorem HasFiniteFPowerSeriesOnBall.neg (hf : HasFiniteFPowerSeriesOnBall f pf x n r) :
    HasFiniteFPowerSeriesOnBall (-f) (-pf) x n r :=
                           /-
                             𝕜 : Type u_1
                             E : Type u_2
                             F : Type u_3
                             inst✝⁴ : NontriviallyNormedField 𝕜
                             inst✝³ : NormedAddCommGroup E
                             inst✝² : NormedSpace 𝕜 E
                             inst✝¹ : NormedAddCommGroup F
                             inst✝ : NormedSpace 𝕜 F
                             f : E → F
                             pf : FormalMultilinearSeries 𝕜 E F
                             x : E
                             r : ENNReal
                             n : Nat
                             hf : HasFiniteFPowerSeriesOnBall f pf x n r
                             m : Nat
                             hm : LE.le n m
                             ⊢ Eq (Neg.neg pf m) 0
                           -/
  ⟨hf.1.neg, fun m hm ↦ by rw [Pi.neg_apply, hf.finite m hm, neg_zero]⟩
                           /-
                             🎉 no goals
                           -/


theorem HasFiniteFPowerSeriesAt.neg (hf : HasFiniteFPowerSeriesAt f pf x n) :
    HasFiniteFPowerSeriesAt (-f) (-pf) x n :=
  let ⟨_, hrf⟩ := hf
  hrf.neg.hasFiniteFPowerSeriesAt


theorem CPolynomialAt.neg (hf : CPolynomialAt 𝕜 f x) : CPolynomialAt 𝕜 (-f) x :=
  let ⟨_, _, hpf⟩ := hf
  hpf.neg.cPolynomialAt


theorem HasFiniteFPowerSeriesOnBall.sub (hf : HasFiniteFPowerSeriesOnBall f pf x n r)
    (hg : HasFiniteFPowerSeriesOnBall g pg x m r) :
    HasFiniteFPowerSeriesOnBall (f - g) (pf - pg) x (max n m) r := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    n m : Nat
    hf : HasFiniteFPowerSeriesOnBall f pf x n r
    hg : HasFiniteFPowerSeriesOnBall g pg x m r
    ⊢ HasFiniteFPowerSeriesOnBall (HSub.hSub f g) (HSub.hSub pf pg) x (Max.max n m …
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem HasFiniteFPowerSeriesAt.sub (hf : HasFiniteFPowerSeriesAt f pf x n)
    (hg : HasFiniteFPowerSeriesAt g pg x m) :
    HasFiniteFPowerSeriesAt (f - g) (pf - pg) x (max n m) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    pf pg : FormalMultilinearSeries 𝕜 E F
    x : E
    n m : Nat
    hf : HasFiniteFPowerSeriesAt f pf x n
    hg : HasFiniteFPowerSeriesAt g pg x m
    ⊢ HasFiniteFPowerSeriesAt (HSub.hSub f g) (HSub.hSub pf pg) x (Max.max n m)
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem CPolynomialAt.sub (hf : CPolynomialAt 𝕜 f x) (hg : CPolynomialAt 𝕜 g x) :
    CPolynomialAt 𝕜 (f - g) x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f g : E → F
    x : E
    hf : CPolynomialAt 𝕜 f x
    hg : CPolynomialAt 𝕜 g x
    ⊢ CPolynomialAt 𝕜 (HSub.hSub f g) x
  -/
  simpa only [sub_eq_add_neg] using hf.add hg.neg
  /-
    🎉 no goals
  -/


theorem CPolynomialOn.mono {s t : Set E} (hf : CPolynomialOn 𝕜 f t) (hst : s ⊆ t) :
    CPolynomialOn 𝕜 f s :=
  fun z hz => hf z (hst hz)


theorem CPolynomialOn.congr' {s : Set E} (hf : CPolynomialOn 𝕜 f s) (hg : f =ᶠ[𝓝ˢ s] g) :
    CPolynomialOn 𝕜 g s :=
  fun z hz => (hf z hz).congr (mem_nhdsSet_iff_forall.mp hg z hz)


theorem CPolynomialOn_congr' {s : Set E} (h : f =ᶠ[𝓝ˢ s] g) :
    CPolynomialOn 𝕜 f s ↔ CPolynomialOn 𝕜 g s :=
  ⟨fun hf => hf.congr' h, fun hg => hg.congr' h.symm⟩


theorem CPolynomialOn.congr {s : Set E} (hs : IsOpen s) (hf : CPolynomialOn 𝕜 f s)
    (hg : s.EqOn f g) : CPolynomialOn 𝕜 g s :=
  hf.congr' <| mem_nhdsSet_iff_forall.mpr
    (fun _ hz => eventuallyEq_iff_exists_mem.mpr ⟨s, hs.mem_nhds hz, hg⟩)


theorem CPolynomialOn_congr {s : Set E} (hs : IsOpen s) (h : s.EqOn f g) :
    CPolynomialOn 𝕜 f s ↔ CPolynomialOn 𝕜 g s :=
  ⟨fun hf => hf.congr hs h, fun hg => hg.congr hs h.symm⟩


theorem CPolynomialOn.add {s : Set E} (hf : CPolynomialOn 𝕜 f s) (hg : CPolynomialOn 𝕜 g s) :
    CPolynomialOn 𝕜 (f + g) s :=
  fun z hz => (hf z hz).add (hg z hz)


theorem CPolynomialOn.sub {s : Set E} (hf : CPolynomialOn 𝕜 f s) (hg : CPolynomialOn 𝕜 g s) :
    CPolynomialOn 𝕜 (f - g) s :=
  fun z hz => (hf z hz).sub (hg z hz)


/-- If a function `f` has a finite power series `p` on a ball and `g` is a continuous linear map,
then `g ∘ f` has the finite power series `g ∘ p` on the same ball. -/
theorem ContinuousLinearMap.comp_hasFiniteFPowerSeriesOnBall (g : F →L[𝕜] G)
    (h : HasFiniteFPowerSeriesOnBall f p x n r) :
    HasFiniteFPowerSeriesOnBall (g ∘ f) (g.compFormalMultilinearSeries p) x n r :=
  ⟨g.comp_hasFPowerSeriesOnBall h.1, fun m hm ↦ by
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      r : ENNReal
      n : Nat
      g : ContinuousLinearMap (RingHom.id 𝕜) F G
      h : HasFiniteFPowerSeriesOnBall f p x n r
      m : Nat
      hm : LE.le n m
      ⊢ Eq (g.compFormalMultilinearSeries p m) 0
    -/
    rw [compFormalMultilinearSeries_apply, h.finite m hm]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      G : Type u_4
      inst✝⁶ : NontriviallyNormedField 𝕜
      inst✝⁵ : NormedAddCommGroup E
      inst✝⁴ : NormedSpace 𝕜 E
      inst✝³ : NormedAddCommGroup F
      inst✝² : NormedSpace 𝕜 F
      inst✝¹ : NormedAddCommGroup G
      inst✝ : NormedSpace 𝕜 G
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      r : ENNReal
      n : Nat
      g : ContinuousLinearMap (RingHom.id 𝕜) F G
      h : HasFiniteFPowerSeriesOnBall f p x n r
      m : Nat
      hm : LE.le n m
      ⊢ Eq (g.compContinuousMultilinearMap 0) 0
    -/
    ext; exact map_zero g⟩
         /-
           🎉 no goals
         -/


/-- If a function `f` is continuously polynomial on a set `s` and `g` is a continuous linear map,
then `g ∘ f` is continuously polynomial on `s`. -/
theorem ContinuousLinearMap.comp_cPolynomialOn {s : Set E} (g : F →L[𝕜] G)
    (h : CPolynomialOn 𝕜 f s) : CPolynomialOn 𝕜 (g ∘ f) s := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → F
    s : Set E
    g : ContinuousLinearMap (RingHom.id 𝕜) F G
    h : CPolynomialOn 𝕜 f s
    ⊢ CPolynomialOn 𝕜 (Function.comp (⇑g) f) s
  -/
  rintro x hx
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → F
    s : Set E
    g : ContinuousLinearMap (RingHom.id 𝕜) F G
    h : CPolynomialOn 𝕜 f s
    x : E
    hx : Membership.mem s x
    ⊢ CPolynomialAt 𝕜 (Function.comp (⇑g) f) x
  -/
  rcases h x hx with ⟨p, n, r, hp⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    G : Type u_4
    inst✝⁶ : NontriviallyNormedField 𝕜
    inst✝⁵ : NormedAddCommGroup E
    inst✝⁴ : NormedSpace 𝕜 E
    inst✝³ : NormedAddCommGroup F
    inst✝² : NormedSpace 𝕜 F
    inst✝¹ : NormedAddCommGroup G
    inst✝ : NormedSpace 𝕜 G
    f : E → F
    s : Set E
    g : ContinuousLinearMap (RingHom.id 𝕜) F G
    h : CPolynomialOn 𝕜 f s
    x : E
    hx : Membership.mem s x
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    r : ENNReal
    hp : HasFiniteFPowerSeriesOnBall f p x n r
    ⊢ CPolynomialAt 𝕜 (Function.comp (⇑g) f) x
  -/
  exact ⟨g.compFormalMultilinearSeries p, n, r, g.comp_hasFiniteFPowerSeriesOnBall hp⟩
  /-
    🎉 no goals
  -/


/-- If a function admits a finite power series expansion bounded by `n`, then it is equal to
the `m`th partial sums of this power series at every point of the disk for `n ≤ m`. -/
theorem HasFiniteFPowerSeriesOnBall.eq_partialSum
    (hf : HasFiniteFPowerSeriesOnBall f p x n r) :
    ∀ y ∈ EMetric.ball (0 : E) r, ∀ m, n ≤ m →
    f (x + y) = p.partialSum m y :=
  fun y hy m hm ↦ (hf.hasSum hy).unique (hasSum_sum_of_ne_finset_zero
    (f := fun m => p m (fun _ => y)) (s := Finset.range m)
                    /-
                      𝕜 : Type u_1
                      E : Type u_2
                      F : Type u_3
                      inst✝⁴ : NontriviallyNormedField 𝕜
                      inst✝³ : NormedAddCommGroup E
                      inst✝² : NormedSpace 𝕜 E
                      inst✝¹ : NormedAddCommGroup F
                      inst✝ : NormedSpace 𝕜 F
                      f : E → F
                      p : FormalMultilinearSeries 𝕜 E F
                      x : E
                      r : ENNReal
                      n : Nat
                      hf : HasFiniteFPowerSeriesOnBall f p x n r
                      y : E
                      hy : Membership.mem (EMetric.ball 0 r) y
                      m : Nat
                      hm : LE.le n m
                      N : Nat
                      hN : Not (Membership.mem (Finset.range m) N)
                      ⊢ Eq ((fun m => (p m) fun x => y) N) 0
                    -/
    (fun N hN => by simp only; simp only [Finset.mem_range, not_lt] at hN
                    /-
                      𝕜 : Type u_1
                      E : Type u_2
                      F : Type u_3
                      inst✝⁴ : NontriviallyNormedField 𝕜
                      inst✝³ : NormedAddCommGroup E
                      inst✝² : NormedSpace 𝕜 E
                      inst✝¹ : NormedAddCommGroup F
                      inst✝ : NormedSpace 𝕜 F
                      f : E → F
                      p : FormalMultilinearSeries 𝕜 E F
                      x : E
                      r : ENNReal
                      n : Nat
                      hf : HasFiniteFPowerSeriesOnBall f p x n r
                      y : E
                      hy : Membership.mem (EMetric.ball 0 r) y
                      m : Nat
                      hm : LE.le n m
                      N : Nat
                      hN : LE.le m N
                      ⊢ Eq ((p N) fun x => y) 0
                    -/
                    rw [hf.finite _ (le_trans hm hN), ContinuousMultilinearMap.zero_apply]))
                    /-
                      🎉 no goals
                    -/


/-- Variant of the previous result with the variable expressed as `y` instead of `x + y`. -/
theorem HasFiniteFPowerSeriesOnBall.eq_partialSum'
    (hf : HasFiniteFPowerSeriesOnBall f p x n r) :
    ∀ y ∈ EMetric.ball x r, ∀ m, n ≤ m →
    f y = p.partialSum m (y - x) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    n : Nat
    hf : HasFiniteFPowerSeriesOnBall f p x n r
    ⊢ ∀ (y : E), Membership.mem (EMetric.ball x r) y → ∀ (m : Nat), LE.le n m → Eq …
  -/
  intro y hy m hm
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    n : Nat
    hf : HasFiniteFPowerSeriesOnBall f p x n r
    y : E
    hy : Membership.mem (EMetric.ball x r) y
    m : Nat
    hm : LE.le n m
    ⊢ Eq (f y) (p.partialSum m (HSub.hSub y x))
  -/
  rw [EMetric.mem_ball, edist_eq_coe_nnnorm_sub, ← mem_emetric_ball_zero_iff] at hy
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    n : Nat
    hf : HasFiniteFPowerSeriesOnBall f p x n r
    y : E
    hy : Membership.mem (EMetric.ball 0 r) (HSub.hSub y x)
    m : Nat
    hm : LE.le n m
    ⊢ Eq (f y) (p.partialSum m (HSub.hSub y x))
  -/
  rw [← (HasFiniteFPowerSeriesOnBall.eq_partialSum hf _ hy m hm), add_sub_cancel]
  /-
    🎉 no goals
  -/


/-- If `f` has a formal power series on a ball bounded by `0`, then `f` is equal to `0` on
the ball. -/
theorem HasFiniteFPowerSeriesOnBall.eq_zero_of_bound_zero
    (hf : HasFiniteFPowerSeriesOnBall f pf x 0 r) : ∀ y ∈ EMetric.ball x r, f y = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    pf : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFiniteFPowerSeriesOnBall f pf x 0 r
    ⊢ ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) 0
  -/
  intro y hy
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    pf : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFiniteFPowerSeriesOnBall f pf x 0 r
    y : E
    hy : Membership.mem (EMetric.ball x r) y
    ⊢ Eq (f y) 0
  -/
  rw [hf.eq_partialSum' y hy 0 le_rfl, FormalMultilinearSeries.partialSum]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    pf : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFiniteFPowerSeriesOnBall f pf x 0 r
    y : E
    hy : Membership.mem (EMetric.ball x r) y
    ⊢ Eq ((Finset.range 0).sum fun k => (pf k) fun x_1 => HSub.hSub y x) 0
  -/
  simp only [Finset.range_zero, Finset.sum_empty]
  /-
    🎉 no goals
  -/


theorem HasFiniteFPowerSeriesOnBall.bound_zero_of_eq_zero (hf : ∀ y ∈ EMetric.ball x r, f y = 0)
    (r_pos : 0 < r) (hp : ∀ n, p n = 0) : HasFiniteFPowerSeriesOnBall f p x 0 r := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) 0
    r_pos : LT.lt 0 r
    hp : ∀ (n : Nat), Eq (p n) 0
    ⊢ HasFiniteFPowerSeriesOnBall f p x 0 r
  -/
  refine ⟨⟨?_, r_pos, ?_⟩, fun n _ ↦ hp n⟩
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      r : ENNReal
      hf : ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) 0
      r_pos : LT.lt 0 r
      hp : ∀ (n : Nat), Eq (p n) 0
      ⊢ LE.le r p.radius
    -/
  · rw [p.radius_eq_top_of_forall_image_add_eq_zero 0 (fun n ↦ by rw [add_zero]; exact hp n)]
    /-
      case refine_1
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      r : ENNReal
      hf : ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) 0
      r_pos : LT.lt 0 r
      hp : ∀ (n : Nat), Eq (p n) 0
      ⊢ LE.le r Top.top
    -/
    exact le_top
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      r : ENNReal
      hf : ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) 0
      r_pos : LT.lt 0 r
      hp : ∀ (n : Nat), Eq (p n) 0
      ⊢ ∀ {y : E}, Membership.mem (EMetric.ball 0 r) y → HasSum (fun n => (p n) fun  …
    -/
  · intro y hy
    /-
      case refine_2
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      x : E
      r : ENNReal
      hf : ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) 0
      r_pos : LT.lt 0 r
      hp : ∀ (n : Nat), Eq (p n) 0
      y : E
      hy : Membership.mem (EMetric.ball 0 r) y
      ⊢ HasSum (fun n => (p n) fun x => y) (f (HAdd.hAdd x y))
    -/
    rw [hf (x + y)]
      /-
        case refine_2
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        x : E
        r : ENNReal
        hf : ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) 0
        r_pos : LT.lt 0 r
        hp : ∀ (n : Nat), Eq (p n) 0
        y : E
        hy : Membership.mem (EMetric.ball 0 r) y
        ⊢ HasSum (fun n => (p n) fun x => y) 0
      -/
    · convert hasSum_zero
      /-
        case h.e'_5.h
        𝕜 : Type u_1
        E : Type u_2
        F : Type u_3
        inst✝⁴ : NontriviallyNormedField 𝕜
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        p : FormalMultilinearSeries 𝕜 E F
        x : E
        r : ENNReal
        hf : ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) 0
        r_pos : LT.lt 0 r
        hp : ∀ (n : Nat), Eq (p n) 0
        y : E
        hy : Membership.mem (EMetric.ball 0 r) y
        x✝ : Nat
        ⊢ Eq ((p x✝) fun x => y) 0
      -/
      rw [hp, ContinuousMultilinearMap.zero_apply]
      /-
        🎉 no goals
      -/
    · rwa [EMetric.mem_ball, edist_eq_coe_nnnorm_sub, add_comm, add_sub_cancel_right,
        ← edist_eq_coe_nnnorm, ← EMetric.mem_ball]


/-- If `f` has a formal power series at `x` bounded by `0`, then `f` is equal to `0` in a
neighborhood of `x`. -/
theorem HasFiniteFPowerSeriesAt.eventually_zero_of_bound_zero
    (hf : HasFiniteFPowerSeriesAt f pf x 0) : f =ᶠ[𝓝 x] 0 :=
  Filter.eventuallyEq_iff_exists_mem.mpr (let ⟨r, hf⟩ := hf; ⟨EMetric.ball x r,
    EMetric.ball_mem_nhds x hf.r_pos, fun y hy ↦ hf.eq_zero_of_bound_zero y hy⟩)


/-- If `f` has a formal power series on a ball bounded by `1`, then `f` is constant equal
to `f x` on the ball. -/
theorem HasFiniteFPowerSeriesOnBall.eq_const_of_bound_one
    (hf : HasFiniteFPowerSeriesOnBall f pf x 1 r) : ∀ y ∈ EMetric.ball x r, f y = f x := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    pf : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFiniteFPowerSeriesOnBall f pf x 1 r
    ⊢ ∀ (y : E), Membership.mem (EMetric.ball x r) y → Eq (f y) (f x)
  -/
  intro y hy
  rw [hf.eq_partialSum' y hy 1 le_rfl, hf.eq_partialSum' x
    (by rw [EMetric.mem_ball, edist_self]; exact hf.r_pos) 1 le_rfl]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    pf : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFiniteFPowerSeriesOnBall f pf x 1 r
    y : E
    hy : Membership.mem (EMetric.ball x r) y
    ⊢ Eq (pf.partialSum 1 (HSub.hSub y x)) (pf.partialSum 1 (HSub.hSub x x))
  -/
  simp only [FormalMultilinearSeries.partialSum, Finset.range_one, Finset.sum_singleton]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    pf : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFiniteFPowerSeriesOnBall f pf x 1 r
    y : E
    hy : Membership.mem (EMetric.ball x r) y
    ⊢ Eq ((pf 0) fun x_1 => HSub.hSub y x) ((pf 0) fun x_1 => HSub.hSub x x)
  -/
  congr
  /-
    case h.e_6.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    pf : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFiniteFPowerSeriesOnBall f pf x 1 r
    y : E
    hy : Membership.mem (EMetric.ball x r) y
    ⊢ Eq (fun x_1 => HSub.hSub y x) fun x_1 => HSub.hSub x x
  -/
  apply funext
  /-
    case h.e_6.h.h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    pf : FormalMultilinearSeries 𝕜 E F
    x : E
    r : ENNReal
    hf : HasFiniteFPowerSeriesOnBall f pf x 1 r
    y : E
    hy : Membership.mem (EMetric.ball x r) y
    ⊢ Fin 0 → Eq (HSub.hSub y x) (HSub.hSub x x)
  -/
  simp only [IsEmpty.forall_iff]
  /-
    🎉 no goals
  -/


/-- If `f` has a formal power series at x bounded by `1`, then `f` is constant equal
to `f x` in a neighborhood of `x`. -/
theorem HasFiniteFPowerSeriesAt.eventually_const_of_bound_one
    (hf : HasFiniteFPowerSeriesAt f pf x 1) : f =ᶠ[𝓝 x] (fun _ => f x) :=
  Filter.eventuallyEq_iff_exists_mem.mpr (let ⟨r, hf⟩ := hf; ⟨EMetric.ball x r,
    EMetric.ball_mem_nhds x hf.r_pos, fun y hy ↦ hf.eq_const_of_bound_one y hy⟩)


/-- If a function admits a finite power series expansion on a disk, then it is continuous there. -/
protected theorem HasFiniteFPowerSeriesOnBall.continuousOn
    (hf : HasFiniteFPowerSeriesOnBall f p x n r) :
    ContinuousOn f (EMetric.ball x r) := hf.1.continuousOn


protected theorem HasFiniteFPowerSeriesAt.continuousAt (hf : HasFiniteFPowerSeriesAt f p x n) :
    ContinuousAt f x := hf.toHasFPowerSeriesAt.continuousAt


protected theorem CPolynomialAt.continuousAt (hf : CPolynomialAt 𝕜 f x) : ContinuousAt f x :=
  hf.analyticAt.continuousAt


protected theorem CPolynomialOn.continuousOn {s : Set E} (hf : CPolynomialOn 𝕜 f s) :
    ContinuousOn f s :=
  hf.analyticOnNhd.continuousOn


/-- Continuously polynomial everywhere implies continuous -/
theorem CPolynomialOn.continuous {f : E → F} (fa : CPolynomialOn 𝕜 f univ) : Continuous f := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    fa : CPolynomialOn 𝕜 f Set.univ
    ⊢ Continuous f
  -/
  rw [continuous_iff_continuousOn_univ]; exact fa.continuousOn
                                         /-
                                           🎉 no goals
                                         -/


protected theorem FormalMultilinearSeries.sum_of_finite (p : FormalMultilinearSeries 𝕜 E F)
    {n : ℕ} (hn : ∀ m, n ≤ m → p m = 0) (x : E) :
    p.sum x = p.partialSum n x :=
                            /-
                              𝕜 : Type u_1
                              E : Type u_2
                              F : Type u_3
                              inst✝⁴ : NontriviallyNormedField 𝕜
                              inst✝³ : NormedAddCommGroup E
                              inst✝² : NormedSpace 𝕜 E
                              inst✝¹ : NormedAddCommGroup F
                              inst✝ : NormedSpace 𝕜 F
                              p : FormalMultilinearSeries 𝕜 E F
                              n : Nat
                              hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
                              x : E
                              m : Nat
                              hm : Not (Membership.mem (Finset.range n) m)
                              ⊢ Eq ((p m) fun x_1 => x) 0
                            -/
  tsum_eq_sum fun m hm ↦ by rw [Finset.mem_range, not_lt] at hm; rw [hn m hm]; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- A finite formal multilinear series sums to its sum at every point. -/
protected theorem FormalMultilinearSeries.hasSum_of_finite (p : FormalMultilinearSeries 𝕜 E F)
    {n : ℕ} (hn : ∀ m, n ≤ m → p m = 0) (x : E) :
    HasSum (fun n : ℕ => p n fun _ => x) (p.sum x) :=
  summable_of_ne_finset_zero (s := .range n)
                   /-
                     𝕜 : Type u_1
                     E : Type u_2
                     F : Type u_3
                     inst✝⁴ : NontriviallyNormedField 𝕜
                     inst✝³ : NormedAddCommGroup E
                     inst✝² : NormedSpace 𝕜 E
                     inst✝¹ : NormedAddCommGroup F
                     inst✝ : NormedSpace 𝕜 F
                     p : FormalMultilinearSeries 𝕜 E F
                     n : Nat
                     hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
                     x : E
                     m : Nat
                     hm : Not (Membership.mem (Finset.range n) m)
                     ⊢ Eq ((p m) fun x_1 => x) 0
                   -/
    (fun m hm ↦ by rw [Finset.mem_range, not_lt] at hm; rw [hn m hm]; rfl)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/
    |>.hasSum


/-- The sum of a finite power series `p` admits `p` as a power series. -/
protected theorem FormalMultilinearSeries.hasFiniteFPowerSeriesOnBall_of_finite
    (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ} (hn : ∀ m, n ≤ m → p m = 0) :
    HasFiniteFPowerSeriesOnBall p.sum p 0 n ⊤ where
             /-
               𝕜 : Type u_1
               E : Type u_2
               F : Type u_3
               inst✝⁴ : NontriviallyNormedField 𝕜
               inst✝³ : NormedAddCommGroup E
               inst✝² : NormedSpace 𝕜 E
               inst✝¹ : NormedAddCommGroup F
               inst✝ : NormedSpace 𝕜 F
               p : FormalMultilinearSeries 𝕜 E F
               n : Nat
               hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
               ⊢ LE.le Top.top p.radius
             -/
  r_le := by rw [radius_eq_top_of_forall_image_add_eq_zero p n fun _ => hn _ (Nat.le_add_left _ _)]
             /-
               🎉 no goals
             -/
  r_pos := zero_lt_top
  finite := hn
                     /-
                       𝕜 : Type u_1
                       E : Type u_2
                       F : Type u_3
                       inst✝⁴ : NontriviallyNormedField 𝕜
                       inst✝³ : NormedAddCommGroup E
                       inst✝² : NormedSpace 𝕜 E
                       inst✝¹ : NormedAddCommGroup F
                       inst✝ : NormedSpace 𝕜 F
                       p : FormalMultilinearSeries 𝕜 E F
                       n : Nat
                       hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
                       y : E
                       x✝ : Membership.mem (EMetric.ball 0 Top.top) y
                       ⊢ HasSum (fun n => (p n) fun x => y) (p.sum (HAdd.hAdd 0 y))
                     -/
  hasSum {y} _ := by rw [zero_add]; exact p.hasSum_of_finite hn y
                                    /-
                                      🎉 no goals
                                    -/


theorem HasFiniteFPowerSeriesOnBall.sum (h : HasFiniteFPowerSeriesOnBall f p x n r) {y : E}
    (hy : y ∈ EMetric.ball (0 : E) r) : f (x + y) = p.sum y :=
  (h.hasSum hy).tsum_eq.symm


/-- The sum of a finite power series is continuous. -/
protected theorem FormalMultilinearSeries.continuousOn_of_finite
    (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ} (hn : ∀ m, n ≤ m → p m = 0) :
    Continuous p.sum := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    ⊢ Continuous p.sum
  -/
  rw [continuous_iff_continuousOn_univ, ← Metric.emetric_ball_top]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    ⊢ ContinuousOn p.sum (EMetric.ball ?m.314284 Top.top)
  -/
  exact (p.hasFiniteFPowerSeriesOnBall_of_finite hn).continuousOn
  /-
    🎉 no goals
  -/


/-- If `p` is a formal multilinear series such that `p m = 0` for `n ≤ m`, then
`p.changeOriginSeriesTerm k l = 0` for `n ≤ k + l`. -/
lemma changeOriginSeriesTerm_bound (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ}
    (hn : ∀ (m : ℕ), n ≤ m → p m = 0) (k l : ℕ) {s : Finset (Fin (k + l))}
    (hs : s.card = l) (hkl : n ≤ k + l) :
    p.changeOriginSeriesTerm k l s hs = 0 := by
  #adaptation_note
  /-- `set_option maxSynthPendingDepth 2` required after https://github.com/leanprover/lean4/pull/4119 -/
  set_option maxSynthPendingDepth 2 in
  rw [changeOriginSeriesTerm, hn _ hkl, map_zero]


/-- If `p` is a finite formal multilinear series, then so is `p.changeOriginSeries k` for every
`k` in `ℕ`. More precisely, if `p m = 0` for `n ≤ m`, then `p.changeOriginSeries k m = 0` for
`n ≤ k + m`. -/
lemma changeOriginSeries_finite_of_finite (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ}
    (hn : ∀ (m : ℕ), n ≤ m → p m = 0) (k : ℕ) : ∀ {m : ℕ}, n ≤ k + m →
    p.changeOriginSeries k m = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k : Nat
    ⊢ ∀ {m : Nat}, LE.le n (HAdd.hAdd k m) → Eq (p.changeOriginSeries k m) 0
  -/
  intro m hm
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k m : Nat
    hm : LE.le n (HAdd.hAdd k m)
    ⊢ Eq (p.changeOriginSeries k m) 0
  -/
  rw [changeOriginSeries]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k m : Nat
    hm : LE.le n (HAdd.hAdd k m)
    ⊢ Eq (Finset.univ.sum fun s => p.changeOriginSeriesTerm k m ↑s ⋯) 0
  -/
  exact Finset.sum_eq_zero (fun _ _ => p.changeOriginSeriesTerm_bound hn _ _ _ hm)
  /-
    🎉 no goals
  -/


lemma changeOriginSeries_sum_eq_partialSum_of_finite (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ}
    (hn : ∀ (m : ℕ), n ≤ m → p m = 0) (k : ℕ) :
    (p.changeOriginSeries k).sum = (p.changeOriginSeries k).partialSum (n - k) := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k : Nat
    ⊢ Eq (p.changeOriginSeries k).sum ((p.changeOriginSeries k).partialSum (HSub.h …
  -/
  ext x
  rw [partialSum, FormalMultilinearSeries.sum,
    tsum_eq_sum (f := fun m => p.changeOriginSeries k m (fun _ => x)) (s := Finset.range (n - k))]
  /-
    case h.H
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k : Nat
    x : E
    x✝ : Fin k → E
    ⊢ ∀ (b : Nat), Not (Membership.mem (Finset.range (HSub.hSub n k)) b) → Eq ((p. …
  -/
  intro m hm
  /-
    case h.H
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k : Nat
    x : E
    x✝ : Fin k → E
    m : Nat
    hm : Not (Membership.mem (Finset.range (HSub.hSub n k)) m)
    ⊢ Eq ((p.changeOriginSeries k m) fun x_1 => x) 0
  -/
  rw [Finset.mem_range, not_lt] at hm
  rw [p.changeOriginSeries_finite_of_finite hn k (by rw [add_comm]; exact Nat.le_add_of_sub_le hm),
    ContinuousMultilinearMap.zero_apply]


/-- If `p` is a formal multilinear series such that `p m = 0` for `n ≤ m`, then
`p.changeOrigin x k = 0` for `n ≤ k`. -/
lemma changeOrigin_finite_of_finite (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ}
    (hn : ∀ (m : ℕ), n ≤ m → p m = 0) {k : ℕ} (hk : n ≤ k) :
    p.changeOrigin x k = 0 := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k : Nat
    hk : LE.le n k
    ⊢ Eq (p.changeOrigin x k) 0
  -/
  rw [changeOrigin, p.changeOriginSeries_sum_eq_partialSum_of_finite hn]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k : Nat
    hk : LE.le n k
    ⊢ Eq ((p.changeOriginSeries k).partialSum (HSub.hSub n k) x) 0
  -/
  apply Finset.sum_eq_zero
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k : Nat
    hk : LE.le n k
    ⊢ ∀ (x_1 : Nat), Membership.mem (Finset.range (HSub.hSub n k)) x_1 → Eq ((p.ch …
  -/
  intro m hm
  /-
    case h
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    x : E
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    k : Nat
    hk : LE.le n k
    m : Nat
    hm : Membership.mem (Finset.range (HSub.hSub n k)) m
    ⊢ Eq ((p.changeOriginSeries k m) fun x_1 => x) 0
  -/
  rw [Finset.mem_range] at hm
  rw [p.changeOriginSeries_finite_of_finite hn k (le_add_of_le_left hk),
    ContinuousMultilinearMap.zero_apply]


theorem hasFiniteFPowerSeriesOnBall_changeOrigin (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ}
    (k : ℕ) (hn : ∀ (m : ℕ), n + k ≤ m → p m = 0) :
    HasFiniteFPowerSeriesOnBall (p.changeOrigin · k) (p.changeOriginSeries k) 0 n ⊤ :=
  (p.changeOriginSeries k).hasFiniteFPowerSeriesOnBall_of_finite
    (fun _ hm => p.changeOriginSeries_finite_of_finite hn k
        /-
          𝕜 : Type u_1
          E : Type u_2
          F : Type u_3
          inst✝⁴ : NontriviallyNormedField 𝕜
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          p : FormalMultilinearSeries 𝕜 E F
          n k : Nat
          hn : ∀ (m : Nat), LE.le (HAdd.hAdd n k) m → Eq (p m) 0
          x✝ : Nat
          hm : LE.le n x✝
          ⊢ LE.le (HAdd.hAdd n k) (HAdd.hAdd k x✝)
        -/
    (by rw [add_comm n k]; apply add_le_add_left hm))
                           /-
                             🎉 no goals
                           -/


theorem changeOrigin_eval_of_finite (p : FormalMultilinearSeries 𝕜 E F) {n : ℕ}
    (hn : ∀ (m : ℕ), n ≤ m → p m = 0) (x y : E) :
    (p.changeOrigin x).sum y = p.sum (x + y) := by
  let f (s : Σ k l : ℕ, { s : Finset (Fin (k + l)) // s.card = l }) : F :=
    p.changeOriginSeriesTerm s.1 s.2.1 s.2.2 s.2.2.2 (fun _ ↦ x) fun _ ↦ y
  have finsupp : f.support.Finite := by
    apply Set.Finite.subset (s := changeOriginIndexEquiv ⁻¹' (Sigma.fst ⁻¹' {m | m < n}))
    · apply Set.Finite.preimage (Equiv.injective _).injOn
      simp_rw [← {m | m < n}.iUnion_of_singleton_coe, preimage_iUnion, ← range_sigmaMk]
      exact finite_iUnion fun _ ↦ finite_range _
    · refine fun s ↦ Not.imp_symm fun hs ↦ ?_
      simp only [preimage_setOf_eq, changeOriginIndexEquiv_apply_fst, mem_setOf, not_lt] at hs
      dsimp only [f]
      rw [changeOriginSeriesTerm_bound p hn _ _ _ hs, ContinuousMultilinearMap.zero_apply,
        ContinuousMultilinearMap.zero_apply]
  have hfkl k l : HasSum (f ⟨k, l, ·⟩) (changeOriginSeries p k l (fun _ ↦ x) fun _ ↦ y) := by
    simp_rw [changeOriginSeries, ContinuousMultilinearMap.sum_apply]; apply hasSum_fintype
  have hfk k : HasSum (f ⟨k, ·⟩) (changeOrigin p x k fun _ ↦ y) := by
    have (m) (hm : m ∉ Finset.range n) : changeOriginSeries p k m (fun _ ↦ x) = 0 := by
      rw [Finset.mem_range, not_lt] at hm
      rw [changeOriginSeries_finite_of_finite _ hn _ (le_add_of_le_right hm),
        ContinuousMultilinearMap.zero_apply]
    rw [changeOrigin, FormalMultilinearSeries.sum,
      ContinuousMultilinearMap.tsum_eval (summable_of_ne_finset_zero this)]
    refine (summable_of_ne_finset_zero (s := Finset.range n) fun m hm ↦ ?_).hasSum.sigma_of_hasSum
      (hfkl k) (summable_of_finite_support <| finsupp.preimage sigma_mk_injective.injOn)
    rw [this m hm, ContinuousMultilinearMap.zero_apply]
  have hf : HasSum f ((p.changeOrigin x).sum y) :=
    ((p.changeOrigin x).hasSum_of_finite (fun _ ↦ changeOrigin_finite_of_finite p hn) _)
      |>.sigma_of_hasSum hfk (summable_of_finite_support finsupp)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    hn : ∀ (m : Nat), LE.le n m → Eq (p m) 0
    x y : E
    f : (Sigma fun k => Sigma fun l => Subtype fun s => Eq s.card l) → F := fun s  …
    finsupp : (Function.support f).Finite
    hfkl : ∀ (k l : Nat), HasSum (fun x => f ⟨k, ⟨l, x⟩⟩) (((p.changeOriginSeries  …
    hfk : ∀ (k : Nat), HasSum (fun x => f ⟨k, x⟩) ((p.changeOrigin x k) fun x => y)
    hf : HasSum f ((p.changeOrigin x).sum y)
    ⊢ Eq ((p.changeOrigin x).sum y) (p.sum (HAdd.hAdd x y))
  -/
  refine hf.unique (changeOriginIndexEquiv.symm.hasSum_iff.1 ?_)
  refine (p.hasSum_of_finite hn (x + y)).sigma_of_hasSum (fun n ↦ ?_)
    (changeOriginIndexEquiv.symm.summable_iff.2 hf.summable)
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n✝ : Nat
    hn : ∀ (m : Nat), LE.le n✝ m → Eq (p m) 0
    x y : E
    f : (Sigma fun k => Sigma fun l => Subtype fun s => Eq s.card l) → F := fun s  …
    finsupp : (Function.support f).Finite
    hfkl : ∀ (k l : Nat), HasSum (fun x => f ⟨k, ⟨l, x⟩⟩) (((p.changeOriginSeries  …
    hfk : ∀ (k : Nat), HasSum (fun x => f ⟨k, x⟩) ((p.changeOrigin x k) fun x => y)
    hf : HasSum f ((p.changeOrigin x).sum y)
    n : Nat
    ⊢ HasSum (fun c => Function.comp f ⇑FormalMultilinearSeries.changeOriginIndexE …
  -/
  rw [← Pi.add_def, (p n).map_add_univ (fun _ ↦ x) fun _ ↦ y]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n✝ : Nat
    hn : ∀ (m : Nat), LE.le n✝ m → Eq (p m) 0
    x y : E
    f : (Sigma fun k => Sigma fun l => Subtype fun s => Eq s.card l) → F := fun s  …
    finsupp : (Function.support f).Finite
    hfkl : ∀ (k l : Nat), HasSum (fun x => f ⟨k, ⟨l, x⟩⟩) (((p.changeOriginSeries  …
    hfk : ∀ (k : Nat), HasSum (fun x => f ⟨k, x⟩) ((p.changeOrigin x k) fun x => y)
    hf : HasSum f ((p.changeOrigin x).sum y)
    n : Nat
    ⊢ HasSum (fun c => Function.comp f ⇑FormalMultilinearSeries.changeOriginIndexE …
  -/
  simp_rw [← changeOriginSeriesTerm_changeOriginIndexEquiv_symm]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    p : FormalMultilinearSeries 𝕜 E F
    n✝ : Nat
    hn : ∀ (m : Nat), LE.le n✝ m → Eq (p m) 0
    x y : E
    f : (Sigma fun k => Sigma fun l => Subtype fun s => Eq s.card l) → F := fun s  …
    finsupp : (Function.support f).Finite
    hfkl : ∀ (k l : Nat), HasSum (fun x => f ⟨k, ⟨l, x⟩⟩) (((p.changeOriginSeries  …
    hfk : ∀ (k : Nat), HasSum (fun x => f ⟨k, x⟩) ((p.changeOrigin x k) fun x => y)
    hf : HasSum f ((p.changeOrigin x).sum y)
    n : Nat
    ⊢ HasSum (fun c => Function.comp f ⇑FormalMultilinearSeries.changeOriginIndexE …
  -/
  exact hasSum_fintype fun c ↦ f (changeOriginIndexEquiv.symm ⟨n, c⟩)
  /-
    🎉 no goals
  -/


/-- The terms of the formal multilinear series `p.changeOrigin` are continuously polynomial
as we vary the origin -/
theorem cPolynomialAt_changeOrigin_of_finite (p : FormalMultilinearSeries 𝕜 E F)
    {n : ℕ} (hn : ∀ (m : ℕ), n ≤ m → p m = 0) (k : ℕ) :
    CPolynomialAt 𝕜 (p.changeOrigin · k) 0 :=
  (p.hasFiniteFPowerSeriesOnBall_changeOrigin k fun _ h ↦ hn _ (le_self_add.trans h)).cPolynomialAt


theorem HasFiniteFPowerSeriesOnBall.changeOrigin (hf : HasFiniteFPowerSeriesOnBall f p x n r)
    (h : (‖y‖₊ : ℝ≥0∞) < r) :
    HasFiniteFPowerSeriesOnBall f (p.changeOrigin y) (x + y) n (r - ‖y‖₊) where
  r_le := (tsub_le_tsub_right hf.r_le _).trans p.changeOrigin_radius
              /-
                𝕜 : Type u_1
                E : Type u_2
                F : Type u_3
                inst✝⁴ : NontriviallyNormedField 𝕜
                inst✝³ : NormedAddCommGroup E
                inst✝² : NormedSpace 𝕜 E
                inst✝¹ : NormedAddCommGroup F
                inst✝ : NormedSpace 𝕜 F
                f : E → F
                p : FormalMultilinearSeries 𝕜 E F
                r : ENNReal
                n : Nat
                x y : E
                hf : HasFiniteFPowerSeriesOnBall f p x n r
                h : LT.lt (↑(NNNorm.nnnorm y)) r
                ⊢ LT.lt 0 (HSub.hSub r ↑(NNNorm.nnnorm y))
              -/
  r_pos := by simp [h]
              /-
                🎉 no goals
              -/
  finite _ hm := p.changeOrigin_finite_of_finite hf.finite hm
  hasSum {z} hz := by
    have : f (x + y + z) =
        FormalMultilinearSeries.sum (FormalMultilinearSeries.changeOrigin p y) z := by
      rw [mem_emetric_ball_zero_iff, lt_tsub_iff_right, add_comm] at hz
      rw [p.changeOrigin_eval_of_finite hf.finite, add_assoc, hf.sum]
      refine mem_emetric_ball_zero_iff.2 (lt_of_le_of_lt ?_ hz)
      exact mod_cast nnnorm_add_le y z
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      n : Nat
      x y : E
      hf : HasFiniteFPowerSeriesOnBall f p x n r
      h : LT.lt (↑(NNNorm.nnnorm y)) r
      z : E
      hz : Membership.mem (EMetric.ball 0 (HSub.hSub r ↑(NNNorm.nnnorm y))) z
      this : Eq (f (HAdd.hAdd (HAdd.hAdd x y) z)) ((p.changeOrigin y).sum z)
      ⊢ HasSum (fun n => (p.changeOrigin y n) fun x => z) (f (HAdd.hAdd (HAdd.hAdd x …
    -/
    rw [this]
    /-
      𝕜 : Type u_1
      E : Type u_2
      F : Type u_3
      inst✝⁴ : NontriviallyNormedField 𝕜
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      p : FormalMultilinearSeries 𝕜 E F
      r : ENNReal
      n : Nat
      x y : E
      hf : HasFiniteFPowerSeriesOnBall f p x n r
      h : LT.lt (↑(NNNorm.nnnorm y)) r
      z : E
      hz : Membership.mem (EMetric.ball 0 (HSub.hSub r ↑(NNNorm.nnnorm y))) z
      this : Eq (f (HAdd.hAdd (HAdd.hAdd x y) z)) ((p.changeOrigin y).sum z)
      ⊢ HasSum (fun n => (p.changeOrigin y n) fun x => z) ((p.changeOrigin y).sum z)
    -/
    apply (p.changeOrigin y).hasSum_of_finite fun _ => p.changeOrigin_finite_of_finite hf.finite
    /-
      🎉 no goals
    -/


/-- If a function admits a finite power series expansion `p` on an open ball `B (x, r)`, then
it is continuously polynomial at every point of this ball. -/
theorem HasFiniteFPowerSeriesOnBall.cPolynomialAt_of_mem
    (hf : HasFiniteFPowerSeriesOnBall f p x n r) (h : y ∈ EMetric.ball x r) :
    CPolynomialAt 𝕜 f y := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    n : Nat
    x y : E
    hf : HasFiniteFPowerSeriesOnBall f p x n r
    h : Membership.mem (EMetric.ball x r) y
    ⊢ CPolynomialAt 𝕜 f y
  -/
  have : (‖y - x‖₊ : ℝ≥0∞) < r := by simpa [edist_eq_coe_nnnorm_sub] using h
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    n : Nat
    x y : E
    hf : HasFiniteFPowerSeriesOnBall f p x n r
    h : Membership.mem (EMetric.ball x r) y
    this : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
    ⊢ CPolynomialAt 𝕜 f y
  -/
  have := hf.changeOrigin this
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    n : Nat
    x y : E
    hf : HasFiniteFPowerSeriesOnBall f p x n r
    h : Membership.mem (EMetric.ball x r) y
    this✝ : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
    this : HasFiniteFPowerSeriesOnBall f (p.changeOrigin (HSub.hSub y x)) (HAdd.hA …
    ⊢ CPolynomialAt 𝕜 f y
  -/
  rw [add_sub_cancel] at this
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    p : FormalMultilinearSeries 𝕜 E F
    r : ENNReal
    n : Nat
    x y : E
    hf : HasFiniteFPowerSeriesOnBall f p x n r
    h : Membership.mem (EMetric.ball x r) y
    this✝ : LT.lt (↑(NNNorm.nnnorm (HSub.hSub y x))) r
    this : HasFiniteFPowerSeriesOnBall f (p.changeOrigin (HSub.hSub y x)) y n (HSu …
    ⊢ CPolynomialAt 𝕜 f y
  -/
  exact this.cPolynomialAt
  /-
    🎉 no goals
  -/


theorem HasFiniteFPowerSeriesOnBall.cPolynomialOn (hf : HasFiniteFPowerSeriesOnBall f p x n r) :
    CPolynomialOn 𝕜 f (EMetric.ball x r) :=
  fun _y hy => hf.cPolynomialAt_of_mem hy


/-- For any function `f` from a normed vector space to a normed vector space, the set of points
`x` such that `f` is continuously polynomial at `x` is open. -/
theorem isOpen_cPolynomialAt : IsOpen { x | CPolynomialAt 𝕜 f x } := by
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    ⊢ IsOpen (setOf fun x => CPolynomialAt 𝕜 f x)
  -/
  rw [isOpen_iff_mem_nhds]
  /-
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    ⊢ ∀ (x : E), Membership.mem (setOf fun x => CPolynomialAt 𝕜 f x) x → Membershi …
  -/
  rintro x ⟨p, n, r, hr⟩
  /-
    case intro.intro.intro
    𝕜 : Type u_1
    E : Type u_2
    F : Type u_3
    inst✝⁴ : NontriviallyNormedField 𝕜
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    p : FormalMultilinearSeries 𝕜 E F
    n : Nat
    r : ENNReal
    hr : HasFiniteFPowerSeriesOnBall f p x n r
    ⊢ Membership.mem (nhds x) (setOf fun x => CPolynomialAt 𝕜 f x)
  -/
  exact mem_of_superset (EMetric.ball_mem_nhds _ hr.r_pos) fun y hy => hr.cPolynomialAt_of_mem hy
  /-
    🎉 no goals
  -/


theorem CPolynomialAt.eventually_cPolynomialAt {f : E → F} {x : E} (h : CPolynomialAt 𝕜 f x) :
    ∀ᶠ y in 𝓝 x, CPolynomialAt 𝕜 f y :=
  (isOpen_cPolynomialAt 𝕜 f).mem_nhds h


theorem CPolynomialAt.exists_mem_nhds_cPolynomialOn {f : E → F} {x : E} (h : CPolynomialAt 𝕜 f x) :
    ∃ s ∈ 𝓝 x, CPolynomialOn 𝕜 f s :=
  h.eventually_cPolynomialAt.exists_mem


/-- If `f` is continuously polynomial at a point, then it is continuously polynomial in a
nonempty ball around that point. -/
theorem CPolynomialAt.exists_ball_cPolynomialOn {f : E → F} {x : E} (h : CPolynomialAt 𝕜 f x) :
    ∃ r : ℝ, 0 < r ∧ CPolynomialOn 𝕜 f (Metric.ball x r) :=
  Metric.isOpen_iff.mp (isOpen_cPolynomialAt _ _) _ h


protected theorem hasFiniteFPowerSeriesOnBall :
    HasFiniteFPowerSeriesOnBall f f.toFormalMultilinearSeries 0 (Fintype.card ι + 1) ⊤ :=
  .mk' (fun _ hm ↦ dif_neg (Nat.succ_le_iff.mp hm).ne) ENNReal.zero_lt_top fun y _ ↦ by
    /-
      𝕜 : Type u_1
      F : Type u_3
      inst✝⁵ : NontriviallyNormedField 𝕜
      inst✝⁴ : NormedAddCommGroup F
      inst✝³ : NormedSpace 𝕜 F
      ι : Type u_5
      Em : ι → Type u_6
      inst✝² : (i : ι) → NormedAddCommGroup (Em i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
      inst✝ : Fintype ι
      f : ContinuousMultilinearMap 𝕜 Em F
      y : (i : ι) → Em i
      x✝ : Membership.mem (EMetric.ball 0 Top.top) y
      ⊢ Eq ((Finset.range (HAdd.hAdd (Fintype.card ι) 1)).sum fun i => (f.toFormalMu …
    -/
    rw [Finset.sum_eq_single_of_mem _ (Finset.self_mem_range_succ _), zero_add]
      /-
        𝕜 : Type u_1
        F : Type u_3
        inst✝⁵ : NontriviallyNormedField 𝕜
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        ι : Type u_5
        Em : ι → Type u_6
        inst✝² : (i : ι) → NormedAddCommGroup (Em i)
        inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
        inst✝ : Fintype ι
        f : ContinuousMultilinearMap 𝕜 Em F
        y : (i : ι) → Em i
        x✝ : Membership.mem (EMetric.ball 0 Top.top) y
        ⊢ Eq ((f.toFormalMultilinearSeries (Fintype.card ι)) fun x => y) (f y)
      -/
    · rw [toFormalMultilinearSeries, dif_pos rfl]; rfl
                                                   /-
                                                     🎉 no goals
                                                   -/
      /-
        𝕜 : Type u_1
        F : Type u_3
        inst✝⁵ : NontriviallyNormedField 𝕜
        inst✝⁴ : NormedAddCommGroup F
        inst✝³ : NormedSpace 𝕜 F
        ι : Type u_5
        Em : ι → Type u_6
        inst✝² : (i : ι) → NormedAddCommGroup (Em i)
        inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
        inst✝ : Fintype ι
        f : ContinuousMultilinearMap 𝕜 Em F
        y : (i : ι) → Em i
        x✝ : Membership.mem (EMetric.ball 0 Top.top) y
        ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd (Fintype.card ι) 1)) b  …
      -/
    · intro m _ ne; rw [toFormalMultilinearSeries, dif_neg ne.symm]; rfl
                                                                     /-
                                                                       🎉 no goals
                                                                     -/


lemma cpolynomialAt  : CPolynomialAt 𝕜 f x :=
  f.hasFiniteFPowerSeriesOnBall.cPolynomialAt_of_mem
        /-
          𝕜 : Type u_1
          F : Type u_3
          inst✝⁵ : NontriviallyNormedField 𝕜
          inst✝⁴ : NormedAddCommGroup F
          inst✝³ : NormedSpace 𝕜 F
          ι : Type u_5
          Em : ι → Type u_6
          inst✝² : (i : ι) → NormedAddCommGroup (Em i)
          inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
          inst✝ : Fintype ι
          f : ContinuousMultilinearMap 𝕜 Em F
          x : (i : ι) → Em i
          ⊢ Membership.mem (EMetric.ball 0 Top.top) x
        -/
    (by simp only [Metric.emetric_ball_top, Set.mem_univ])
        /-
          🎉 no goals
        -/


lemma cpolyomialOn : CPolynomialOn 𝕜 f s := fun _ _ ↦ f.cpolynomialAt


lemma analyticOnNhd : AnalyticOnNhd 𝕜 f s := f.cpolyomialOn.analyticOnNhd


lemma analyticOn : AnalyticOn 𝕜 f s := f.analyticOnNhd.analyticOn


@[deprecated (since := "2024-09-26")]
alias analyticWithinOn := analyticOn


lemma analyticAt : AnalyticAt 𝕜 f x := f.cpolynomialAt.analyticAt


lemma analyticWithinAt : AnalyticWithinAt 𝕜 f s x := f.analyticAt.analyticWithinAt


/-- Formal multilinear series associated to a linear map into multilinear maps. -/
noncomputable def toFormalMultilinearSeriesOfMultilinear :
    FormalMultilinearSeries 𝕜 (G × (Π i, Em i)) F :=
  fun n ↦ if h : Fintype.card (Option ι) = n then
    (f.continuousMultilinearMapOption).domDomCongr (Fintype.equivFinOfCardEq h)
  else 0


protected theorem hasFiniteFPowerSeriesOnBall_uncurry_of_multilinear :
    HasFiniteFPowerSeriesOnBall (fun (p : G × (Π i, Em i)) ↦ f p.1 p.2)
      f.toFormalMultilinearSeriesOfMultilinear 0 (Fintype.card (Option ι) + 1) ⊤ := by
  /-
    𝕜 : Type u_1
    F : Type u_3
    G : Type u_4
    inst✝⁷ : NontriviallyNormedField 𝕜
    inst✝⁶ : NormedAddCommGroup F
    inst✝⁵ : NormedSpace 𝕜 F
    inst✝⁴ : NormedAddCommGroup G
    inst✝³ : NormedSpace 𝕜 G
    ι : Type u_5
    Em : ι → Type u_6
    inst✝² : (i : ι) → NormedAddCommGroup (Em i)
    inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
    inst✝ : Fintype ι
    f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
    ⊢ HasFiniteFPowerSeriesOnBall (fun p => (f p.1) p.2) f.toFormalMultilinearSeri …
  -/
  apply HasFiniteFPowerSeriesOnBall.mk' ?_ ENNReal.zero_lt_top  ?_
    /-
      𝕜 : Type u_1
      F : Type u_3
      G : Type u_4
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      ι : Type u_5
      Em : ι → Type u_6
      inst✝² : (i : ι) → NormedAddCommGroup (Em i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
      inst✝ : Fintype ι
      f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
      ⊢ ∀ (m : Nat), LE.le (HAdd.hAdd (Fintype.card (Option ι)) 1) m → Eq (f.toForma …
    -/
  · intro m hm
    /-
      𝕜 : Type u_1
      F : Type u_3
      G : Type u_4
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      ι : Type u_5
      Em : ι → Type u_6
      inst✝² : (i : ι) → NormedAddCommGroup (Em i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
      inst✝ : Fintype ι
      f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
      m : Nat
      hm : LE.le (HAdd.hAdd (Fintype.card (Option ι)) 1) m
      ⊢ Eq (f.toFormalMultilinearSeriesOfMultilinear m) 0
    -/
    apply dif_neg
    /-
      case hnc
      𝕜 : Type u_1
      F : Type u_3
      G : Type u_4
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      ι : Type u_5
      Em : ι → Type u_6
      inst✝² : (i : ι) → NormedAddCommGroup (Em i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
      inst✝ : Fintype ι
      f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
      m : Nat
      hm : LE.le (HAdd.hAdd (Fintype.card (Option ι)) 1) m
      ⊢ Not (Eq (Fintype.card (Option ι)) m)
    -/
    exact Nat.ne_of_lt hm
    /-
      🎉 no goals
    -/
    /-
      𝕜 : Type u_1
      F : Type u_3
      G : Type u_4
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      ι : Type u_5
      Em : ι → Type u_6
      inst✝² : (i : ι) → NormedAddCommGroup (Em i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
      inst✝ : Fintype ι
      f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
      ⊢ ∀ (y : Prod G ((i : ι) → Em i)), Membership.mem (EMetric.ball 0 Top.top) y → …
    -/
  · intro y _
    /-
      𝕜 : Type u_1
      F : Type u_3
      G : Type u_4
      inst✝⁷ : NontriviallyNormedField 𝕜
      inst✝⁶ : NormedAddCommGroup F
      inst✝⁵ : NormedSpace 𝕜 F
      inst✝⁴ : NormedAddCommGroup G
      inst✝³ : NormedSpace 𝕜 G
      ι : Type u_5
      Em : ι → Type u_6
      inst✝² : (i : ι) → NormedAddCommGroup (Em i)
      inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
      inst✝ : Fintype ι
      f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
      y : Prod G ((i : ι) → Em i)
      a✝ : Membership.mem (EMetric.ball 0 Top.top) y
      ⊢ Eq ((Finset.range (HAdd.hAdd (Fintype.card (Option ι)) 1)).sum fun i => (f.t …
    -/
    rw [Finset.sum_eq_single_of_mem _ (Finset.self_mem_range_succ _), zero_add]
      /-
        𝕜 : Type u_1
        F : Type u_3
        G : Type u_4
        inst✝⁷ : NontriviallyNormedField 𝕜
        inst✝⁶ : NormedAddCommGroup F
        inst✝⁵ : NormedSpace 𝕜 F
        inst✝⁴ : NormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        ι : Type u_5
        Em : ι → Type u_6
        inst✝² : (i : ι) → NormedAddCommGroup (Em i)
        inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
        inst✝ : Fintype ι
        f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
        y : Prod G ((i : ι) → Em i)
        a✝ : Membership.mem (EMetric.ball 0 Top.top) y
        ⊢ Eq ((f.toFormalMultilinearSeriesOfMultilinear (Fintype.card (Option ι))) fun …
      -/
    · rw [toFormalMultilinearSeriesOfMultilinear, dif_pos rfl]; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/
      /-
        𝕜 : Type u_1
        F : Type u_3
        G : Type u_4
        inst✝⁷ : NontriviallyNormedField 𝕜
        inst✝⁶ : NormedAddCommGroup F
        inst✝⁵ : NormedSpace 𝕜 F
        inst✝⁴ : NormedAddCommGroup G
        inst✝³ : NormedSpace 𝕜 G
        ι : Type u_5
        Em : ι → Type u_6
        inst✝² : (i : ι) → NormedAddCommGroup (Em i)
        inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
        inst✝ : Fintype ι
        f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
        y : Prod G ((i : ι) → Em i)
        a✝ : Membership.mem (EMetric.ball 0 Top.top) y
        ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd (Fintype.card (Option ι …
      -/
    · intro m _ ne; rw [toFormalMultilinearSeriesOfMultilinear, dif_neg ne.symm]; rfl
                                                                                  /-
                                                                                    🎉 no goals
                                                                                  -/


lemma cpolynomialAt_uncurry_of_multilinear :
    CPolynomialAt 𝕜 (fun (p : G × (Π i, Em i)) ↦ f p.1 p.2) x :=
  f.hasFiniteFPowerSeriesOnBall_uncurry_of_multilinear.cPolynomialAt_of_mem
        /-
          𝕜 : Type u_1
          F : Type u_3
          G : Type u_4
          inst✝⁷ : NontriviallyNormedField 𝕜
          inst✝⁶ : NormedAddCommGroup F
          inst✝⁵ : NormedSpace 𝕜 F
          inst✝⁴ : NormedAddCommGroup G
          inst✝³ : NormedSpace 𝕜 G
          ι : Type u_5
          Em : ι → Type u_6
          inst✝² : (i : ι) → NormedAddCommGroup (Em i)
          inst✝¹ : (i : ι) → NormedSpace 𝕜 (Em i)
          inst✝ : Fintype ι
          f : ContinuousLinearMap (RingHom.id 𝕜) G (ContinuousMultilinearMap 𝕜 Em F)
          x : Prod G ((i : ι) → Em i)
          ⊢ Membership.mem (EMetric.ball 0 Top.top) x
        -/
    (by simp only [Metric.emetric_ball_top, Set.mem_univ])
        /-
          🎉 no goals
        -/


lemma cpolyomialOn_uncurry_of_multilinear :
    CPolynomialOn 𝕜 (fun (p : G × (Π i, Em i)) ↦ f p.1 p.2) s :=
  fun _ _ ↦ f.cpolynomialAt_uncurry_of_multilinear


lemma analyticOnNhd_uncurry_of_multilinear :
    AnalyticOnNhd 𝕜 (fun (p : G × (Π i, Em i)) ↦ f p.1 p.2) s :=
  f.cpolyomialOn_uncurry_of_multilinear.analyticOnNhd


lemma analyticOn_uncurry_of_multilinear :
    AnalyticOn 𝕜 (fun (p : G × (Π i, Em i)) ↦ f p.1 p.2) s :=
  f.analyticOnNhd_uncurry_of_multilinear.analyticOn


@[deprecated (since := "2024-09-26")]
alias analyticWithinOn_uncurry_of_multilinear := analyticOn_uncurry_of_multilinear


lemma analyticAt_uncurry_of_multilinear : AnalyticAt 𝕜 (fun (p : G × (Π i, Em i)) ↦ f p.1 p.2) x :=
  f.cpolynomialAt_uncurry_of_multilinear.analyticAt


lemma analyticWithinAt_uncurry_of_multilinear :
    AnalyticWithinAt 𝕜 (fun (p : G × (Π i, Em i)) ↦ f p.1 p.2) s x :=
  f.analyticAt_uncurry_of_multilinear.analyticWithinAt


