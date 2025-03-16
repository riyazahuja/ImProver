/-- Smoothness exponent for analytic functions. -/
scoped [ContDiff] notation3 "ω" => (⊤ : WithTop ℕ∞)

/-- Smoothness exponent for infinitely differentiable functions. -/
scoped [ContDiff] notation3 "∞" => ((⊤ : ℕ∞) : WithTop ℕ∞)


/-- `HasFTaylorSeriesUpToOn n f p s` registers the fact that `p 0 = f` and `p (m+1)` is a
derivative of `p m` for `m < n`, and is continuous for `m ≤ n`. This is a predicate analogous to
`HasFDerivWithinAt` but for higher order derivatives.

Notice that `p` does not sum up to `f` on the diagonal (`FormalMultilinearSeries.sum`), even if
`f` is analytic and `n = ∞`: an additional `1/m!` factor on the `m`th term is necessary for that. -/
structure HasFTaylorSeriesUpToOn
  (n : WithTop ℕ∞) (f : E → F) (p : E → FormalMultilinearSeries 𝕜 E F) (s : Set E) : Prop where
  zero_eq : ∀ x ∈ s, (p x 0).curry0 = f x
  protected fderivWithin : ∀ m : ℕ, m < n → ∀ x ∈ s,
    HasFDerivWithinAt (p · m) (p x m.succ).curryLeft s x
  cont : ∀ m : ℕ, m ≤ n → ContinuousOn (p · m) s


theorem HasFTaylorSeriesUpToOn.zero_eq' (h : HasFTaylorSeriesUpToOn n f p s) {x : E} (hx : x ∈ s) :
    p x 0 = (continuousMultilinearCurryFin0 𝕜 E F).symm (f x) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    x : E
    hx : Membership.mem s x
    ⊢ Eq (p x 0) ((continuousMultilinearCurryFin0 𝕜 E F).symm (f x))
  -/
  rw [← h.zero_eq x hx]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    x : E
    hx : Membership.mem s x
    ⊢ Eq (p x 0) ((continuousMultilinearCurryFin0 𝕜 E F).symm (p x 0).curry0)
  -/
  exact (p x 0).uncurry0_curry0.symm
  /-
    🎉 no goals
  -/


/-- If two functions coincide on a set `s`, then a Taylor series for the first one is as well a
Taylor series for the second one. -/
theorem HasFTaylorSeriesUpToOn.congr (h : HasFTaylorSeriesUpToOn n f p s)
    (h₁ : ∀ x ∈ s, f₁ x = f x) : HasFTaylorSeriesUpToOn n f₁ p s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f f₁ : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    h₁ : ∀ (x : E), Membership.mem s x → Eq (f₁ x) (f x)
    ⊢ HasFTaylorSeriesUpToOn n f₁ p s
  -/
  refine ⟨fun x hx => ?_, h.fderivWithin, h.cont⟩
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f f₁ : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    h₁ : ∀ (x : E), Membership.mem s x → Eq (f₁ x) (f x)
    x : E
    hx : Membership.mem s x
    ⊢ Eq (p x 0).curry0 (f₁ x)
  -/
  rw [h₁ x hx]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f f₁ : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    h₁ : ∀ (x : E), Membership.mem s x → Eq (f₁ x) (f x)
    x : E
    hx : Membership.mem s x
    ⊢ Eq (p x 0).curry0 (f x)
  -/
  exact h.zero_eq x hx
  /-
    🎉 no goals
  -/


theorem HasFTaylorSeriesUpToOn.congr_series {q} (hp : HasFTaylorSeriesUpToOn n f p s)
    (hpq : ∀ m : ℕ, m ≤ n → EqOn (p · m) (q · m) s) :
    HasFTaylorSeriesUpToOn n f q s where
                     /-
                       𝕜 : Type u
                       inst✝⁴ : NontriviallyNormedField 𝕜
                       E : Type uE
                       inst✝³ : NormedAddCommGroup E
                       inst✝² : NormedSpace 𝕜 E
                       F : Type uF
                       inst✝¹ : NormedAddCommGroup F
                       inst✝ : NormedSpace 𝕜 F
                       s : Set E
                       f : E → F
                       n : WithTop ENat
                       p q : E → FormalMultilinearSeries 𝕜 E F
                       hp : HasFTaylorSeriesUpToOn n f p s
                       hpq : ∀ (m : Nat), LE.le (↑m) n → Set.EqOn (fun x => p x m) (fun x => q x m) s
                       x : E
                       hx : Membership.mem s x
                       ⊢ Eq (q x 0).curry0 (f x)
                     -/
  zero_eq x hx := by simp only [← (hpq 0 (zero_le n) hx), hp.zero_eq x hx]
                     /-
                       🎉 no goals
                     -/
  fderivWithin m hm x hx := by
    /-
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      n : WithTop ENat
      p q : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn n f p s
      hpq : ∀ (m : Nat), LE.le (↑m) n → Set.EqOn (fun x => p x m) (fun x => q x m) s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ HasFDerivWithinAt (fun x => q x m) (q x m.succ).curryLeft s x
    -/
    refine ((hp.fderivWithin m hm x hx).congr' (hpq m hm.le).symm hx).congr_fderiv ?_
    /-
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      n : WithTop ENat
      p q : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn n f p s
      hpq : ∀ (m : Nat), LE.le (↑m) n → Set.EqOn (fun x => p x m) (fun x => q x m) s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ Eq (p x m.succ).curryLeft (q x m.succ).curryLeft
    -/
    refine congrArg _ (hpq (m + 1) ?_ hx)
    /-
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      n : WithTop ENat
      p q : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn n f p s
      hpq : ∀ (m : Nat), LE.le (↑m) n → Set.EqOn (fun x => p x m) (fun x => q x m) s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ LE.le (↑(HAdd.hAdd m 1)) n
    -/
    exact ENat.add_one_natCast_le_withTop_of_lt hm
    /-
      🎉 no goals
    -/
  cont m hm := (hp.cont m hm).congr (hpq m hm).symm


theorem HasFTaylorSeriesUpToOn.mono (h : HasFTaylorSeriesUpToOn n f p s) {t : Set E} (hst : t ⊆ s) :
    HasFTaylorSeriesUpToOn n f p t :=
  ⟨fun x hx => h.zero_eq x (hst hx), fun m hm x hx => (h.fderivWithin m hm x (hst hx)).mono hst,
    fun m hm => (h.cont m hm).mono hst⟩


theorem HasFTaylorSeriesUpToOn.of_le (h : HasFTaylorSeriesUpToOn n f p s) (hmn : m ≤ n) :
    HasFTaylorSeriesUpToOn m f p s :=
  ⟨h.zero_eq, fun k hk x hx => h.fderivWithin k (lt_of_lt_of_le hk hmn) x hx, fun k hk =>
    h.cont k (le_trans hk hmn)⟩


theorem HasFTaylorSeriesUpToOn.continuousOn (h : HasFTaylorSeriesUpToOn n f p s) :
    ContinuousOn f s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    ⊢ ContinuousOn f s
  -/
  have := (h.cont 0 bot_le).congr fun x hx => (h.zero_eq' hx).symm
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    this : ContinuousOn (fun x => (continuousMultilinearCurryFin0 𝕜 E F).symm (f x …
    ⊢ ContinuousOn f s
  -/
  rwa [← (continuousMultilinearCurryFin0 𝕜 E F).symm.comp_continuousOn_iff]
  /-
    🎉 no goals
  -/


theorem hasFTaylorSeriesUpToOn_zero_iff :
    HasFTaylorSeriesUpToOn 0 f p s ↔ ContinuousOn f s ∧ ∀ x ∈ s, (p x 0).curry0 = f x := by
  refine ⟨fun H => ⟨H.continuousOn, H.zero_eq⟩, fun H =>
      ⟨H.2, fun m hm => False.elim (not_le.2 hm bot_le), fun m hm ↦ ?_⟩⟩
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    p : E → FormalMultilinearSeries 𝕜 E F
    H : And (ContinuousOn f s) (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0  …
    m : Nat
    hm : LE.le (↑m) 0
    ⊢ ContinuousOn (fun x => p x m) s
  -/
  obtain rfl : m = 0 := mod_cast hm.antisymm (zero_le _)
  have : EqOn (p · 0) ((continuousMultilinearCurryFin0 𝕜 E F).symm ∘ f) s := fun x hx ↦
    (continuousMultilinearCurryFin0 𝕜 E F).eq_symm_apply.2 (H.2 x hx)
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    p : E → FormalMultilinearSeries 𝕜 E F
    H : And (ContinuousOn f s) (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0  …
    hm : LE.le (↑0) 0
    this : Set.EqOn (fun x => p x 0) (Function.comp (⇑(continuousMultilinearCurryF …
    ⊢ ContinuousOn (fun x => p x 0) s
  -/
  rw [continuousOn_congr this, LinearIsometryEquiv.comp_continuousOn_iff]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    p : E → FormalMultilinearSeries 𝕜 E F
    H : And (ContinuousOn f s) (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0  …
    hm : LE.le (↑0) 0
    this : Set.EqOn (fun x => p x 0) (Function.comp (⇑(continuousMultilinearCurryF …
    ⊢ ContinuousOn f s
  -/
  exact H.1
  /-
    🎉 no goals
  -/


theorem hasFTaylorSeriesUpToOn_top_iff_add (hN : ∞ ≤ N) (k : ℕ) :
    HasFTaylorSeriesUpToOn N f p s ↔ ∀ n : ℕ, HasFTaylorSeriesUpToOn (n + k : ℕ) f p s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    N : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    hN : LE.le (↑Top.top) N
    k : Nat
    ⊢ Iff (HasFTaylorSeriesUpToOn N f p s) (∀ (n : Nat), HasFTaylorSeriesUpToOn (↑ …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      k : Nat
      ⊢ HasFTaylorSeriesUpToOn N f p s → ∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd …
    -/
  · intro H n
    /-
      case mp
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      k : Nat
      H : HasFTaylorSeriesUpToOn N f p s
      n : Nat
      ⊢ HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n k)) f p s
    -/
    apply H.of_le (natCast_le_of_coe_top_le_withTop hN _)
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      k : Nat
      ⊢ (∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n k)) f p s) → HasFTaylorS …
    -/
  · intro H
    /-
      case mpr
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      k : Nat
      H : ∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n k)) f p s
      ⊢ HasFTaylorSeriesUpToOn N f p s
    -/
    constructor
      /-
        case mpr.zero_eq
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        N : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        hN : LE.le (↑Top.top) N
        k : Nat
        H : ∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n k)) f p s
        ⊢ ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
      -/
    · exact (H 0).zero_eq
      /-
        🎉 no goals
      -/
      /-
        case mpr.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        N : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        hN : LE.le (↑Top.top) N
        k : Nat
        H : ∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n k)) f p s
        ⊢ ∀ (m : Nat), LT.lt (↑m) N → ∀ (x : E), Membership.mem s x → HasFDerivWithinA …
      -/
    · intro m _
      /-
        case mpr.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        N : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        hN : LE.le (↑Top.top) N
        k : Nat
        H : ∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n k)) f p s
        m : Nat
        a✝ : LT.lt (↑m) N
        ⊢ ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun x => p x m) (p x m.su …
      -/
      apply (H m.succ).fderivWithin m (by norm_cast; omega)
      /-
        🎉 no goals
      -/
      /-
        case mpr.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        N : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        hN : LE.le (↑Top.top) N
        k : Nat
        H : ∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n k)) f p s
        ⊢ ∀ (m : Nat), LE.le (↑m) N → ContinuousOn (fun x => p x m) s
      -/
    · intro m _
      /-
        case mpr.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        N : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        hN : LE.le (↑Top.top) N
        k : Nat
        H : ∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n k)) f p s
        m : Nat
        a✝ : LE.le (↑m) N
        ⊢ ContinuousOn (fun x => p x m) s
      -/
      apply (H m).cont m (by simp)
      /-
        🎉 no goals
      -/


theorem hasFTaylorSeriesUpToOn_top_iff (hN : ∞ ≤ N) :
    HasFTaylorSeriesUpToOn N f p s ↔ ∀ n : ℕ, HasFTaylorSeriesUpToOn n f p s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    N : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    hN : LE.le (↑Top.top) N
    ⊢ Iff (HasFTaylorSeriesUpToOn N f p s) (∀ (n : Nat), HasFTaylorSeriesUpToOn (↑ …
  -/
  simpa using hasFTaylorSeriesUpToOn_top_iff_add hN 0
  /-
    🎉 no goals
  -/


/-- In the case that `n = ∞` we don't need the continuity assumption in
`HasFTaylorSeriesUpToOn`. -/
theorem hasFTaylorSeriesUpToOn_top_iff' (hN : ∞ ≤ N) :
    HasFTaylorSeriesUpToOn N f p s ↔
      (∀ x ∈ s, (p x 0).curry0 = f x) ∧
        ∀ m : ℕ, ∀ x ∈ s, HasFDerivWithinAt (fun y => p y m) (p x m.succ).curryLeft s x := by
  -- Everything except for the continuity is trivial:
  refine ⟨fun h => ⟨h.1, fun m => h.2 m (natCast_lt_of_coe_top_le_withTop hN _)⟩, fun h =>
    ⟨h.1, fun m _ => h.2 m, fun m _ x hx =>
      -- The continuity follows from the existence of a derivative:
      (h.2 m x hx).continuousWithinAt⟩⟩


/-- If a function has a Taylor series at order at least `1`, then the term of order `1` of this
series is a derivative of `f`. -/
theorem HasFTaylorSeriesUpToOn.hasFDerivWithinAt (h : HasFTaylorSeriesUpToOn n f p s) (hn : 1 ≤ n)
    (hx : x ∈ s) : HasFDerivWithinAt f (continuousMultilinearCurryFin1 𝕜 E F (p x 1)) s x := by
  have A : ∀ y ∈ s, f y = (continuousMultilinearCurryFin0 𝕜 E F) (p y 0) := fun y hy ↦
    (h.zero_eq y hy).symm
  suffices H : HasFDerivWithinAt (continuousMultilinearCurryFin0 𝕜 E F ∘ (p · 0))
    (continuousMultilinearCurryFin1 𝕜 E F (p x 1)) s x from H.congr A (A x hx)
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    hn : LE.le 1 n
    hx : Membership.mem s x
    A : ∀ (y : E), Membership.mem s y → Eq (f y) ((continuousMultilinearCurryFin0  …
    ⊢ HasFDerivWithinAt (Function.comp ⇑(continuousMultilinearCurryFin0 𝕜 E F) fun …
  -/
  rw [LinearIsometryEquiv.comp_hasFDerivWithinAt_iff']
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    hn : LE.le 1 n
    hx : Membership.mem s x
    A : ∀ (y : E), Membership.mem s y → Eq (f y) ((continuousMultilinearCurryFin0  …
    ⊢ HasFDerivWithinAt (fun x => p x 0) ((↑{ toLinearEquiv := (continuousMultilin …
  -/
  have : ((0 : ℕ) : ℕ∞) < n := zero_lt_one.trans_le hn
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    hn : LE.le 1 n
    hx : Membership.mem s x
    A : ∀ (y : E), Membership.mem s y → Eq (f y) ((continuousMultilinearCurryFin0  …
    this : LT.lt (↑↑0) n
    ⊢ HasFDerivWithinAt (fun x => p x 0) ((↑{ toLinearEquiv := (continuousMultilin …
  -/
  convert h.fderivWithin _ this x hx
  /-
    case h.e'_12.h.h
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    hn : LE.le 1 n
    hx : Membership.mem s x
    A : ∀ (y : E), Membership.mem s y → Eq (f y) ((continuousMultilinearCurryFin0  …
    this : LT.lt (↑↑0) n
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup ContinuousMultilinearMap.instA …
    he✝ : Eq NormedSpace.toModule ContinuousMultilinearMap.instModule
    e_10✝ : Eq UniformSpace.toTopologicalSpace ContinuousMultilinearMap.instTopolo …
    ⊢ Eq ((↑{ toLinearEquiv := (continuousMultilinearCurryFin0 𝕜 E F).symm.toLinea …
  -/
  ext y v
  /-
    case h.e'_12.h.h.h.H
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    hn : LE.le 1 n
    hx : Membership.mem s x
    A : ∀ (y : E), Membership.mem s y → Eq (f y) ((continuousMultilinearCurryFin0  …
    this : LT.lt (↑↑0) n
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup ContinuousMultilinearMap.instA …
    he✝ : Eq NormedSpace.toModule ContinuousMultilinearMap.instModule
    e_10✝ : Eq UniformSpace.toTopologicalSpace ContinuousMultilinearMap.instTopolo …
    y : E
    v : Fin 0 → E
    ⊢ Eq ((((↑{ toLinearEquiv := (continuousMultilinearCurryFin0 𝕜 E F).symm.toLin …
  -/
  change (p x 1) (snoc 0 y) = (p x 1) (cons y v)
  /-
    case h.e'_12.h.h.h.H
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    hn : LE.le 1 n
    hx : Membership.mem s x
    A : ∀ (y : E), Membership.mem s y → Eq (f y) ((continuousMultilinearCurryFin0  …
    this : LT.lt (↑↑0) n
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup ContinuousMultilinearMap.instA …
    he✝ : Eq NormedSpace.toModule ContinuousMultilinearMap.instModule
    e_10✝ : Eq UniformSpace.toTopologicalSpace ContinuousMultilinearMap.instTopolo …
    y : E
    v : Fin 0 → E
    ⊢ Eq ((p x 1) (Fin.snoc 0 y)) ((p x 1) (Fin.cons y v))
  -/
  congr with i
  /-
    case h.e'_12.h.h.h.H.h.e_6.h.h
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    hn : LE.le 1 n
    hx : Membership.mem s x
    A : ∀ (y : E), Membership.mem s y → Eq (f y) ((continuousMultilinearCurryFin0  …
    this : LT.lt (↑↑0) n
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup ContinuousMultilinearMap.instA …
    he✝ : Eq NormedSpace.toModule ContinuousMultilinearMap.instModule
    e_10✝ : Eq UniformSpace.toTopologicalSpace ContinuousMultilinearMap.instTopolo …
    y : E
    v : Fin 0 → E
    i : Fin (HAdd.hAdd 0 1)
    ⊢ Eq (Fin.snoc 0 y i) (Fin.cons y v i)
  -/
  rw [Unique.eq_default (α := Fin 1) i]
  /-
    case h.e'_12.h.h.h.H.h.e_6.h.h
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    hn : LE.le 1 n
    hx : Membership.mem s x
    A : ∀ (y : E), Membership.mem s y → Eq (f y) ((continuousMultilinearCurryFin0  …
    this : LT.lt (↑↑0) n
    e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup ContinuousMultilinearMap.instA …
    he✝ : Eq NormedSpace.toModule ContinuousMultilinearMap.instModule
    e_10✝ : Eq UniformSpace.toTopologicalSpace ContinuousMultilinearMap.instTopolo …
    y : E
    v : Fin 0 → E
    i : Fin (HAdd.hAdd 0 1)
    ⊢ Eq (Fin.snoc 0 y Inhabited.default) (Fin.cons y v Inhabited.default)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem HasFTaylorSeriesUpToOn.differentiableOn (h : HasFTaylorSeriesUpToOn n f p s) (hn : 1 ≤ n) :
    DifferentiableOn 𝕜 f s := fun _x hx => (h.hasFDerivWithinAt hn hx).differentiableWithinAt


/-- If a function has a Taylor series at order at least `1` on a neighborhood of `x`, then the term
of order `1` of this series is a derivative of `f` at `x`. -/
theorem HasFTaylorSeriesUpToOn.hasFDerivAt (h : HasFTaylorSeriesUpToOn n f p s) (hn : 1 ≤ n)
    (hx : s ∈ 𝓝 x) : HasFDerivAt f (continuousMultilinearCurryFin1 𝕜 E F (p x 1)) x :=
  (h.hasFDerivWithinAt hn (mem_of_mem_nhds hx)).hasFDerivAt hx


/-- If a function has a Taylor series at order at least `1` on a neighborhood of `x`, then
in a neighborhood of `x`, the term of order `1` of this series is a derivative of `f`. -/
theorem HasFTaylorSeriesUpToOn.eventually_hasFDerivAt (h : HasFTaylorSeriesUpToOn n f p s)
    (hn : 1 ≤ n) (hx : s ∈ 𝓝 x) :
    ∀ᶠ y in 𝓝 x, HasFDerivAt f (continuousMultilinearCurryFin1 𝕜 E F (p y 1)) y :=
  (eventually_eventually_nhds.2 hx).mono fun _y hy => h.hasFDerivAt hn hy


/-- If a function has a Taylor series at order at least `1` on a neighborhood of `x`, then
it is differentiable at `x`. -/
theorem HasFTaylorSeriesUpToOn.differentiableAt (h : HasFTaylorSeriesUpToOn n f p s) (hn : 1 ≤ n)
    (hx : s ∈ 𝓝 x) : DifferentiableAt 𝕜 f x :=
  (h.hasFDerivAt hn hx).differentiableAt


/-- `p` is a Taylor series of `f` up to `n+1` if and only if `p` is a Taylor series up to `n`, and
`p (n + 1)` is a derivative of `p n`. -/
theorem hasFTaylorSeriesUpToOn_succ_iff_left {n : ℕ} :
    HasFTaylorSeriesUpToOn (n + 1) f p s ↔
      HasFTaylorSeriesUpToOn n f p s ∧
        (∀ x ∈ s, HasFDerivWithinAt (fun y => p y n) (p x n.succ).curryLeft s x) ∧
          ContinuousOn (fun x => p x (n + 1)) s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    p : E → FormalMultilinearSeries 𝕜 E F
    n : Nat
    ⊢ Iff (HasFTaylorSeriesUpToOn (HAdd.hAdd (↑n) 1) f p s) (And (HasFTaylorSeries …
  -/
  constructor
  · exact fun h ↦ ⟨h.of_le (mod_cast Nat.le_succ n),
      h.fderivWithin _ (mod_cast lt_add_one n), h.cont (n + 1) le_rfl⟩
    /-
      case mpr
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      ⊢ And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s x  …
    -/
  · intro h
    /-
      case mpr
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
      ⊢ HasFTaylorSeriesUpToOn (HAdd.hAdd (↑n) 1) f p s
    -/
    constructor
      /-
        case mpr.zero_eq
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
        ⊢ ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
      -/
    · exact h.1.zero_eq
      /-
        🎉 no goals
      -/
      /-
        case mpr.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
        ⊢ ∀ (m : Nat), LT.lt (↑m) (HAdd.hAdd (↑n) 1) → ∀ (x : E), Membership.mem s x → …
      -/
    · intro m hm
      /-
        case mpr.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
        m : Nat
        hm : LT.lt (↑m) (HAdd.hAdd (↑n) 1)
        ⊢ ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun x => p x m) (p x m.su …
      -/
      by_cases h' : m < n
        /-
          case pos
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
          m : Nat
          hm : LT.lt (↑m) (HAdd.hAdd (↑n) 1)
          h' : LT.lt m n
          ⊢ ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun x => p x m) (p x m.su …
        -/
      · exact h.1.fderivWithin m (mod_cast h')
        /-
          🎉 no goals
        -/
        /-
          case neg
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
          m : Nat
          hm : LT.lt (↑m) (HAdd.hAdd (↑n) 1)
          h' : Not (LT.lt m n)
          ⊢ ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun x => p x m) (p x m.su …
        -/
      · have : m = n := Nat.eq_of_lt_succ_of_not_lt (mod_cast hm) h'
        /-
          case neg
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
          m : Nat
          hm : LT.lt (↑m) (HAdd.hAdd (↑n) 1)
          h' : Not (LT.lt m n)
          this : Eq m n
          ⊢ ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun x => p x m) (p x m.su …
        -/
        rw [this]
        /-
          case neg
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
          m : Nat
          hm : LT.lt (↑m) (HAdd.hAdd (↑n) 1)
          h' : Not (LT.lt m n)
          this : Eq m n
          ⊢ ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun x => p x n) (p x n.su …
        -/
        exact h.2.1
        /-
          🎉 no goals
        -/
      /-
        case mpr.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
        ⊢ ∀ (m : Nat), LE.le (↑m) (HAdd.hAdd (↑n) 1) → ContinuousOn (fun x => p x m) s
      -/
    · intro m hm
      /-
        case mpr.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
        m : Nat
        hm : LE.le (↑m) (HAdd.hAdd (↑n) 1)
        ⊢ ContinuousOn (fun x => p x m) s
      -/
      by_cases h' : m ≤ n
        /-
          case pos
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
          m : Nat
          hm : LE.le (↑m) (HAdd.hAdd (↑n) 1)
          h' : LE.le m n
          ⊢ ContinuousOn (fun x => p x m) s
        -/
      · apply h.1.cont m (mod_cast h')
        /-
          🎉 no goals
        -/
        /-
          case neg
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
          m : Nat
          hm : LE.le (↑m) (HAdd.hAdd (↑n) 1)
          h' : Not (LE.le m n)
          ⊢ ContinuousOn (fun x => p x m) s
        -/
      · have : m = n + 1 := le_antisymm (mod_cast hm) (not_le.1 h')
        /-
          case neg
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
          m : Nat
          hm : LE.le (↑m) (HAdd.hAdd (↑n) 1)
          h' : Not (LE.le m n)
          this : Eq m (HAdd.hAdd n 1)
          ⊢ ContinuousOn (fun x => p x m) s
        -/
        rw [this]
        /-
          case neg
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          h : And (HasFTaylorSeriesUpToOn (↑n) f p s) (And (∀ (x : E), Membership.mem s  …
          m : Nat
          hm : LE.le (↑m) (HAdd.hAdd (↑n) 1)
          h' : Not (LE.le m n)
          this : Eq m (HAdd.hAdd n 1)
          ⊢ ContinuousOn (fun x => p x (HAdd.hAdd n 1)) s
        -/
        exact h.2.2
        /-
          🎉 no goals
        -/


set_option maxSynthPendingDepth 2 in
-- Porting note: this was split out from `hasFTaylorSeriesUpToOn_succ_iff_right` to avoid a timeout.
theorem HasFTaylorSeriesUpToOn.shift_of_succ
    {n : ℕ} (H : HasFTaylorSeriesUpToOn (n + 1 : ℕ) f p s) :
    (HasFTaylorSeriesUpToOn n (fun x => continuousMultilinearCurryFin1 𝕜 E F (p x 1))
      (fun x => (p x).shift)) s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    p : E → FormalMultilinearSeries 𝕜 E F
    n : Nat
    H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
    ⊢ HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin1 𝕜 E F) …
  -/
  constructor
    /-
      case zero_eq
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      ⊢ ∀ (x : E), Membership.mem s x → Eq ((p x).shift 0).curry0 ((continuousMultil …
    -/
  · intro x _
    /-
      case zero_eq
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      x : E
      a✝ : Membership.mem s x
      ⊢ Eq ((p x).shift 0).curry0 ((continuousMultilinearCurryFin1 𝕜 E F) (p x 1))
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case fderivWithin
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      ⊢ ∀ (m : Nat), LT.lt ↑m ↑n → ∀ (x : E), Membership.mem s x → HasFDerivWithinAt …
    -/
  · intro m (hm : (m : WithTop ℕ∞) < n) x (hx : x ∈ s)
    have A : (m.succ : WithTop ℕ∞) < n.succ := by
      rw [Nat.cast_lt] at hm ⊢
      exact Nat.succ_lt_succ hm
    change HasFDerivWithinAt (continuousMultilinearCurryRightEquiv' 𝕜 m E F ∘ (p · m.succ))
      (p x m.succ.succ).curryRight.curryLeft s x
    /-
      case fderivWithin
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      m : Nat
      hm : LT.lt ↑m ↑n
      x : E
      hx : Membership.mem s x
      A : LT.lt ↑m.succ ↑n.succ
      ⊢ HasFDerivWithinAt (Function.comp ⇑(continuousMultilinearCurryRightEquiv' 𝕜 m …
    -/
    rw [(continuousMultilinearCurryRightEquiv' 𝕜 m E F).comp_hasFDerivWithinAt_iff']
    /-
      case fderivWithin
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      m : Nat
      hm : LT.lt ↑m ↑n
      x : E
      hx : Membership.mem s x
      A : LT.lt ↑m.succ ↑n.succ
      ⊢ HasFDerivWithinAt (fun x => p x m.succ) ((↑{ toLinearEquiv := (continuousMul …
    -/
    convert H.fderivWithin _ A x hx
    /-
      case h.e'_12.h.h
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      m : Nat
      hm : LT.lt ↑m ↑n
      x : E
      hx : Membership.mem s x
      A : LT.lt ↑m.succ ↑n.succ
      e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup ContinuousMultilinearMap.instA …
      he✝ : Eq NormedSpace.toModule ContinuousMultilinearMap.instModule
      e_10✝ : Eq UniformSpace.toTopologicalSpace ContinuousMultilinearMap.instTopolo …
      ⊢ Eq ((↑{ toLinearEquiv := (continuousMultilinearCurryRightEquiv' 𝕜 m E F).sym …
    -/
    ext y v
    /-
      case h.e'_12.h.h.h.H
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      m : Nat
      hm : LT.lt ↑m ↑n
      x : E
      hx : Membership.mem s x
      A : LT.lt ↑m.succ ↑n.succ
      e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup ContinuousMultilinearMap.instA …
      he✝ : Eq NormedSpace.toModule ContinuousMultilinearMap.instModule
      e_10✝ : Eq UniformSpace.toTopologicalSpace ContinuousMultilinearMap.instTopolo …
      y : E
      v : Fin m.succ → E
      ⊢ Eq ((((↑{ toLinearEquiv := (continuousMultilinearCurryRightEquiv' 𝕜 m E F).s …
    -/
    change p x (m + 2) (snoc (cons y (init v)) (v (last _))) = p x (m + 2) (cons y v)
    /-
      case h.e'_12.h.h.h.H
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      m : Nat
      hm : LT.lt ↑m ↑n
      x : E
      hx : Membership.mem s x
      A : LT.lt ↑m.succ ↑n.succ
      e_8✝ : Eq SeminormedAddCommGroup.toAddCommGroup ContinuousMultilinearMap.instA …
      he✝ : Eq NormedSpace.toModule ContinuousMultilinearMap.instModule
      e_10✝ : Eq UniformSpace.toTopologicalSpace ContinuousMultilinearMap.instTopolo …
      y : E
      v : Fin m.succ → E
      ⊢ Eq ((p x (HAdd.hAdd m 2)) (Fin.snoc (Fin.cons y (Fin.init v)) (v (Fin.last m …
    -/
    rw [← cons_snoc_eq_snoc_cons, snoc_init_self]
    /-
      🎉 no goals
    -/
    /-
      case cont
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      ⊢ ∀ (m : Nat), LE.le ↑m ↑n → ContinuousOn (fun x => (p x).shift m) s
    -/
  · intro m (hm : (m : WithTop ℕ∞) ≤ n)
    suffices A : ContinuousOn (p · (m + 1)) s from
      (continuousMultilinearCurryRightEquiv' 𝕜 m E F).continuous.comp_continuousOn A
    /-
      case cont
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      m : Nat
      hm : LE.le ↑m ↑n
      ⊢ ContinuousOn (fun x => p x (HAdd.hAdd m 1)) s
    -/
    refine H.cont _ ?_
    /-
      case cont
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      m : Nat
      hm : LE.le ↑m ↑n
      ⊢ LE.le ↑(HAdd.hAdd m 1) ↑(HAdd.hAdd n 1)
    -/
    rw [Nat.cast_le] at hm ⊢
    /-
      case cont
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      m : Nat
      hm : LE.le m n
      ⊢ LE.le (HAdd.hAdd m 1) (HAdd.hAdd n 1)
    -/
    exact Nat.succ_le_succ hm
    /-
      🎉 no goals
    -/


/-- `p` is a Taylor series of `f` up to `n+1` if and only if `p.shift` is a Taylor series up to `n`
for `p 1`, which is a derivative of `f`. Version for `n : ℕ`. -/
theorem hasFTaylorSeriesUpToOn_succ_nat_iff_right {n : ℕ} :
    HasFTaylorSeriesUpToOn (n + 1 : ℕ) f p s ↔
      (∀ x ∈ s, (p x 0).curry0 = f x) ∧
        (∀ x ∈ s, HasFDerivWithinAt (fun y => p y 0) (p x 1).curryLeft s x) ∧
          HasFTaylorSeriesUpToOn n (fun x => continuousMultilinearCurryFin1 𝕜 E F (p x 1))
            (fun x => (p x).shift) s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    p : E → FormalMultilinearSeries 𝕜 E F
    n : Nat
    ⊢ Iff (HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s) (And (∀ (x : E), Membe …
  -/
  constructor
    /-
      case mp
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      ⊢ HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s → And (∀ (x : E), Membership …
    -/
  · intro H
    /-
      case mp
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      ⊢ And (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)) (And (∀ (x : E …
    -/
    refine ⟨H.zero_eq, H.fderivWithin 0 (Nat.cast_lt.2 (Nat.succ_pos n)), ?_⟩
    /-
      case mp
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      H : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      ⊢ HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin1 𝕜 E F) …
    -/
    exact H.shift_of_succ
    /-
      🎉 no goals
    -/
    /-
      case mpr
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      ⊢ And (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)) (And (∀ (x : E …
    -/
  · rintro ⟨Hzero_eq, Hfderiv_zero, Htaylor⟩
    /-
      case mpr.intro.intro
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      p : E → FormalMultilinearSeries 𝕜 E F
      n : Nat
      Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
      Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
      Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
      ⊢ HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
    -/
    constructor
      /-
        case mpr.intro.intro.zero_eq
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
        Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
        Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
        ⊢ ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
      -/
    · exact Hzero_eq
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
        Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
        Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
        ⊢ ∀ (m : Nat), LT.lt ↑m ↑(HAdd.hAdd n 1) → ∀ (x : E), Membership.mem s x → Has …
      -/
    · intro m (hm : (m : WithTop ℕ∞) < n.succ) x (hx : x ∈ s)
      /-
        case mpr.intro.intro.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
        Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
        Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
        m : Nat
        hm : LT.lt ↑m ↑n.succ
        x : E
        hx : Membership.mem s x
        ⊢ HasFDerivWithinAt (fun x => p x m) (p x m.succ).curryLeft s x
      -/
      cases' m with m
        /-
          case mpr.intro.intro.fderivWithin.zero
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          x : E
          hx : Membership.mem s x
          hm : LT.lt ↑0 ↑n.succ
          ⊢ HasFDerivWithinAt (fun x => p x 0) (p x (Nat.succ 0)).curryLeft s x
        -/
      · exact Hfderiv_zero x hx
        /-
          🎉 no goals
        -/
      · have A : (m : WithTop ℕ∞) < n := by
          rw [Nat.cast_lt] at hm ⊢
          exact Nat.lt_of_succ_lt_succ hm
        have :
          HasFDerivWithinAt (𝕜 := 𝕜) (continuousMultilinearCurryRightEquiv' 𝕜 m E F ∘ (p · m.succ))
            ((p x).shift m.succ).curryLeft s x := Htaylor.fderivWithin _ A x hx
        rw [LinearIsometryEquiv.comp_hasFDerivWithinAt_iff'
            (f' := ((p x).shift m.succ).curryLeft)] at this
        /-
          case mpr.intro.intro.fderivWithin.succ
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          x : E
          hx : Membership.mem s x
          m : Nat
          hm : LT.lt ↑(HAdd.hAdd m 1) ↑n.succ
          A : LT.lt ↑m ↑n
          this : HasFDerivWithinAt (fun x => p x m.succ) ((↑{ toLinearEquiv := (continuo …
          ⊢ HasFDerivWithinAt (fun x => p x (HAdd.hAdd m 1)) (p x (HAdd.hAdd m 1).succ). …
        -/
        convert this
        /-
          case h.e'_12.h.h
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          x : E
          hx : Membership.mem s x
          m : Nat
          hm : LT.lt ↑(HAdd.hAdd m 1) ↑n.succ
          A : LT.lt ↑m ↑n
          this : HasFDerivWithinAt (fun x => p x m.succ) ((↑{ toLinearEquiv := (continuo …
          e_8✝ : Eq ContinuousMultilinearMap.instAddCommGroup SeminormedAddCommGroup.toA …
          he✝ : Eq ContinuousMultilinearMap.instModule NormedSpace.toModule
          e_10✝ : Eq ContinuousMultilinearMap.instTopologicalSpace UniformSpace.toTopolo …
          ⊢ Eq (p x (HAdd.hAdd m 1).succ).curryLeft ((↑{ toLinearEquiv := (continuousMul …
        -/
        ext y v
        change
          (p x (Nat.succ (Nat.succ m))) (cons y v) =
            (p x m.succ.succ) (snoc (cons y (init v)) (v (last _)))
        /-
          case h.e'_12.h.h.h.H
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          x : E
          hx : Membership.mem s x
          m : Nat
          hm : LT.lt ↑(HAdd.hAdd m 1) ↑n.succ
          A : LT.lt ↑m ↑n
          this : HasFDerivWithinAt (fun x => p x m.succ) ((↑{ toLinearEquiv := (continuo …
          e_8✝ : Eq ContinuousMultilinearMap.instAddCommGroup SeminormedAddCommGroup.toA …
          he✝ : Eq ContinuousMultilinearMap.instModule NormedSpace.toModule
          e_10✝ : Eq ContinuousMultilinearMap.instTopologicalSpace UniformSpace.toTopolo …
          y : E
          v : Fin (HAdd.hAdd m 1) → E
          ⊢ Eq ((p x m.succ.succ) (Fin.cons y v)) ((p x m.succ.succ) (Fin.snoc (Fin.cons …
        -/
        rw [← cons_snoc_eq_snoc_cons, snoc_init_self]
        /-
          🎉 no goals
        -/
      /-
        case mpr.intro.intro.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
        Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
        Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
        ⊢ ∀ (m : Nat), LE.le ↑m ↑(HAdd.hAdd n 1) → ContinuousOn (fun x => p x m) s
      -/
    · intro m (hm : (m : WithTop ℕ∞) ≤ n.succ)
      /-
        case mpr.intro.intro.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        s : Set E
        f : E → F
        p : E → FormalMultilinearSeries 𝕜 E F
        n : Nat
        Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
        Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
        Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
        m : Nat
        hm : LE.le ↑m ↑n.succ
        ⊢ ContinuousOn (fun x => p x m) s
      -/
      cases' m with m
      · have : DifferentiableOn 𝕜 (fun x => p x 0) s := fun x hx =>
          (Hfderiv_zero x hx).differentiableWithinAt
        /-
          case mpr.intro.intro.cont.zero
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          hm : LE.le ↑0 ↑n.succ
          this : DifferentiableOn 𝕜 (fun x => p x 0) s
          ⊢ ContinuousOn (fun x => p x 0) s
        -/
        exact this.continuousOn
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.intro.cont.succ
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          m : Nat
          hm : LE.le ↑(HAdd.hAdd m 1) ↑n.succ
          ⊢ ContinuousOn (fun x => p x (HAdd.hAdd m 1)) s
        -/
      · refine (continuousMultilinearCurryRightEquiv' 𝕜 m E F).comp_continuousOn_iff.mp ?_
        /-
          case mpr.intro.intro.cont.succ
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          m : Nat
          hm : LE.le ↑(HAdd.hAdd m 1) ↑n.succ
          ⊢ ContinuousOn (Function.comp ⇑(continuousMultilinearCurryRightEquiv' 𝕜 m E F) …
        -/
        refine Htaylor.cont _ ?_
        /-
          case mpr.intro.intro.cont.succ
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          m : Nat
          hm : LE.le ↑(HAdd.hAdd m 1) ↑n.succ
          ⊢ LE.le ↑m ↑n
        -/
        rw [Nat.cast_le] at hm ⊢
        /-
          case mpr.intro.intro.cont.succ
          𝕜 : Type u
          inst✝⁴ : NontriviallyNormedField 𝕜
          E : Type uE
          inst✝³ : NormedAddCommGroup E
          inst✝² : NormedSpace 𝕜 E
          F : Type uF
          inst✝¹ : NormedAddCommGroup F
          inst✝ : NormedSpace 𝕜 F
          s : Set E
          f : E → F
          p : E → FormalMultilinearSeries 𝕜 E F
          n : Nat
          Hzero_eq : ∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)
          Hfderiv_zero : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt (fun y => p y …
          Htaylor : HasFTaylorSeriesUpToOn (↑n) (fun x => (continuousMultilinearCurryFin …
          m : Nat
          hm : LE.le (HAdd.hAdd m 1) n.succ
          ⊢ LE.le m n
        -/
        exact Nat.lt_succ_iff.mp hm
        /-
          🎉 no goals
        -/


/-- `p` is a Taylor series of `f` up to `⊤` if and only if `p.shift` is a Taylor series up to `⊤`
for `p 1`, which is a derivative of `f`. -/
theorem hasFTaylorSeriesUpToOn_top_iff_right (hN : ∞ ≤ N) :
    HasFTaylorSeriesUpToOn N f p s ↔
      (∀ x ∈ s, (p x 0).curry0 = f x) ∧
        (∀ x ∈ s, HasFDerivWithinAt (fun y => p y 0) (p x 1).curryLeft s x) ∧
          HasFTaylorSeriesUpToOn N (fun x => continuousMultilinearCurryFin1 𝕜 E F (p x 1))
            (fun x => (p x).shift) s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    N : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    hN : LE.le (↑Top.top) N
    ⊢ Iff (HasFTaylorSeriesUpToOn N f p s) (And (∀ (x : E), Membership.mem s x → E …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      h : HasFTaylorSeriesUpToOn N f p s
      ⊢ And (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)) (And (∀ (x : E …
    -/
  · rw [hasFTaylorSeriesUpToOn_top_iff_add hN 1] at h
    /-
      case refine_1
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      h : ∀ (n : Nat), HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
      ⊢ And (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)) (And (∀ (x : E …
    -/
    rw [hasFTaylorSeriesUpToOn_top_iff hN]
    exact ⟨(hasFTaylorSeriesUpToOn_succ_nat_iff_right.1 (h 1)).1,
      (hasFTaylorSeriesUpToOn_succ_nat_iff_right.1 (h 1)).2.1,
      fun n ↦ (hasFTaylorSeriesUpToOn_succ_nat_iff_right.1 (h n)).2.2⟩
    /-
      case refine_2
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      h : And (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)) (And (∀ (x : …
      ⊢ HasFTaylorSeriesUpToOn N f p s
    -/
  · apply (hasFTaylorSeriesUpToOn_top_iff_add hN 1).2 (fun n ↦ ?_)
    /-
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      h : And (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)) (And (∀ (x : …
      n : Nat
      ⊢ HasFTaylorSeriesUpToOn (↑(HAdd.hAdd n 1)) f p s
    -/
    rw [hasFTaylorSeriesUpToOn_succ_nat_iff_right]
    /-
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      N : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      hN : LE.le (↑Top.top) N
      h : And (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)) (And (∀ (x : …
      n : Nat
      ⊢ And (∀ (x : E), Membership.mem s x → Eq (p x 0).curry0 (f x)) (And (∀ (x : E …
    -/
    exact ⟨h.1, h.2.1, (h.2.2).of_le (m := n) (natCast_le_of_coe_top_le_withTop hN n)⟩
    /-
      🎉 no goals
    -/


/-- `p` is a Taylor series of `f` up to `n+1` if and only if `p.shift` is a Taylor series up to `n`
for `p 1`, which is a derivative of `f`. Version for `n : WithTop ℕ∞`. -/
theorem hasFTaylorSeriesUpToOn_succ_iff_right :
    HasFTaylorSeriesUpToOn (n + 1) f p s ↔
      (∀ x ∈ s, (p x 0).curry0 = f x) ∧
        (∀ x ∈ s, HasFDerivWithinAt (fun y => p y 0) (p x 1).curryLeft s x) ∧
          HasFTaylorSeriesUpToOn n (fun x => continuousMultilinearCurryFin1 𝕜 E F (p x 1))
            (fun x => (p x).shift) s := by
  match n with
  | ⊤ => exact hasFTaylorSeriesUpToOn_top_iff_right (by simp)
  | (⊤ : ℕ∞) => exact hasFTaylorSeriesUpToOn_top_iff_right (by simp)
  | (n : ℕ) => exact hasFTaylorSeriesUpToOn_succ_nat_iff_right


/-- The `n`-th derivative of a function along a set, defined inductively by saying that the `n+1`-th
derivative of `f` is the derivative of the `n`-th derivative of `f` along this set, together with
an uncurrying step to see it as a multilinear map in `n+1` variables..
-/
noncomputable def iteratedFDerivWithin (n : ℕ) (f : E → F) (s : Set E) : E → E[×n]→L[𝕜] F :=
  Nat.recOn n (fun x => ContinuousMultilinearMap.uncurry0 𝕜 E (f x)) fun _ rec x =>
    ContinuousLinearMap.uncurryLeft (fderivWithin 𝕜 rec s x)


/-- Formal Taylor series associated to a function within a set. -/
def ftaylorSeriesWithin (f : E → F) (s : Set E) (x : E) : FormalMultilinearSeries 𝕜 E F := fun n =>
  iteratedFDerivWithin 𝕜 n f s x


@[simp]
theorem iteratedFDerivWithin_zero_apply (m : Fin 0 → E) :
    (iteratedFDerivWithin 𝕜 0 f s x : (Fin 0 → E) → F) m = f x :=
  rfl


theorem iteratedFDerivWithin_zero_eq_comp :
    iteratedFDerivWithin 𝕜 0 f s = (continuousMultilinearCurryFin0 𝕜 E F).symm ∘ f :=
  rfl


@[simp]
theorem norm_iteratedFDerivWithin_zero : ‖iteratedFDerivWithin 𝕜 0 f s x‖ = ‖f x‖ := by
  -- Porting note: added `comp_apply`.
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    ⊢ Eq (Norm.norm (iteratedFDerivWithin 𝕜 0 f s x)) (Norm.norm (f x))
  -/
  rw [iteratedFDerivWithin_zero_eq_comp, comp_apply, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


theorem iteratedFDerivWithin_succ_apply_left {n : ℕ} (m : Fin (n + 1) → E) :
    (iteratedFDerivWithin 𝕜 (n + 1) f s x : (Fin (n + 1) → E) → F) m =
      (fderivWithin 𝕜 (iteratedFDerivWithin 𝕜 n f s) s x : E → E[×n]→L[𝕜] F) (m 0) (tail m) :=
  rfl


/-- Writing explicitly the `n+1`-th derivative as the composition of a currying linear equiv,
and the derivative of the `n`-th derivative. -/
theorem iteratedFDerivWithin_succ_eq_comp_left {n : ℕ} :
    iteratedFDerivWithin 𝕜 (n + 1) f s =
      (continuousMultilinearCurryLeftEquiv 𝕜 (fun _ : Fin (n + 1) => E) F).symm ∘
        fderivWithin 𝕜 (iteratedFDerivWithin 𝕜 n f s) s :=
  rfl


theorem fderivWithin_iteratedFDerivWithin {s : Set E} {n : ℕ} :
    fderivWithin 𝕜 (iteratedFDerivWithin 𝕜 n f s) s =
      (continuousMultilinearCurryLeftEquiv 𝕜 (fun _ : Fin (n + 1) => E) F) ∘
        iteratedFDerivWithin 𝕜 (n + 1) f s :=
  rfl


theorem norm_fderivWithin_iteratedFDerivWithin {n : ℕ} :
    ‖fderivWithin 𝕜 (iteratedFDerivWithin 𝕜 n f s) s x‖ =
      ‖iteratedFDerivWithin 𝕜 (n + 1) f s x‖ := by
  -- Porting note: added `comp_apply`.
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : Nat
    ⊢ Eq (Norm.norm (fderivWithin 𝕜 (iteratedFDerivWithin 𝕜 n f s) s x)) (Norm.nor …
  -/
  rw [iteratedFDerivWithin_succ_eq_comp_left, comp_apply, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


theorem iteratedFDerivWithin_succ_apply_right {n : ℕ} (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s)
    (m : Fin (n + 1) → E) :
    (iteratedFDerivWithin 𝕜 (n + 1) f s x : (Fin (n + 1) → E) → F) m =
      iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 f s y) s x (init m) (m (last n)) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : Nat
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    m : Fin (HAdd.hAdd n 1) → E
    ⊢ Eq ((iteratedFDerivWithin 𝕜 (HAdd.hAdd n 1) f s x) m) (((iteratedFDerivWithi …
  -/
  induction' n with n IH generalizing x
  · rw [iteratedFDerivWithin_succ_eq_comp_left, iteratedFDerivWithin_zero_eq_comp,
      iteratedFDerivWithin_zero_apply, Function.comp_apply,
      LinearIsometryEquiv.comp_fderivWithin _ (hs x hx)]
    /-
      case zero
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      hs : UniqueDiffOn 𝕜 s
      x : E
      hx : Membership.mem s x
      m : Fin (HAdd.hAdd 0 1) → E
      ⊢ Eq (((continuousMultilinearCurryLeftEquiv 𝕜 (fun x => E) F).symm ((↑{ toLine …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case succ
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      hs : UniqueDiffOn 𝕜 s
      n : Nat
      IH : ∀ {x : E}, Membership.mem s x → ∀ (m : Fin (HAdd.hAdd n 1) → E), Eq ((ite …
      x : E
      hx : Membership.mem s x
      m : Fin (HAdd.hAdd (HAdd.hAdd n 1) 1) → E
      ⊢ Eq ((iteratedFDerivWithin 𝕜 (HAdd.hAdd (HAdd.hAdd n 1) 1) f s x) m) (((itera …
    -/
  · let I := (continuousMultilinearCurryRightEquiv' 𝕜 n E F).symm
    have A : ∀ y ∈ s, iteratedFDerivWithin 𝕜 n.succ f s y =
        (I ∘ iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 f s y) s) y := fun y hy ↦ by
      ext m
      rw [@IH y hy m]
      rfl
    calc
      (iteratedFDerivWithin 𝕜 (n + 2) f s x : (Fin (n + 2) → E) → F) m =
          (fderivWithin 𝕜 (iteratedFDerivWithin 𝕜 n.succ f s) s x : E → E[×n + 1]→L[𝕜] F) (m 0)
            (tail m) :=
        rfl
      _ = (fderivWithin 𝕜 (I ∘ iteratedFDerivWithin 𝕜 n (fderivWithin 𝕜 f s) s) s x :
              E → E[×n + 1]→L[𝕜] F) (m 0) (tail m) := by
        rw [fderivWithin_congr A (A x hx)]
      _ = (I ∘ fderivWithin 𝕜 (iteratedFDerivWithin 𝕜 n (fderivWithin 𝕜 f s) s) s x :
              E → E[×n + 1]→L[𝕜] F) (m 0) (tail m) := by
        #adaptation_note
        /--
        After https://github.com/leanprover/lean4/pull/4119 we need to either use
        `set_option maxSynthPendingDepth 2 in`
        or fill in an explicit argument as
        ```
        simp only [LinearIsometryEquiv.comp_fderivWithin _
          (f := iteratedFDerivWithin 𝕜 n (fderivWithin 𝕜 f s) s) (hs x hx)]
        ```
        -/
        set_option maxSynthPendingDepth 2 in
          simp only [LinearIsometryEquiv.comp_fderivWithin _ (hs x hx)]
        rfl
      _ = (fderivWithin 𝕜 (iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 f s y) s) s x :
              E → E[×n]→L[𝕜] E →L[𝕜] F) (m 0) (init (tail m)) ((tail m) (last n)) := rfl
      _ = iteratedFDerivWithin 𝕜 (Nat.succ n) (fun y => fderivWithin 𝕜 f s y) s x (init m)
            (m (last (n + 1))) := by
        rw [iteratedFDerivWithin_succ_apply_left, tail_init_eq_init_tail]
        rfl


/-- Writing explicitly the `n+1`-th derivative as the composition of a currying linear equiv,
and the `n`-th derivative of the derivative. -/
theorem iteratedFDerivWithin_succ_eq_comp_right {n : ℕ} (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    iteratedFDerivWithin 𝕜 (n + 1) f s x =
      ((continuousMultilinearCurryRightEquiv' 𝕜 n E F).symm ∘
          iteratedFDerivWithin 𝕜 n (fun y => fderivWithin 𝕜 f s y) s)
        x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : Nat
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ Eq (iteratedFDerivWithin 𝕜 (HAdd.hAdd n 1) f s x) (Function.comp (⇑(continuo …
  -/
  ext m; rw [iteratedFDerivWithin_succ_apply_right hs hx]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem norm_iteratedFDerivWithin_fderivWithin {n : ℕ} (hs : UniqueDiffOn 𝕜 s) (hx : x ∈ s) :
    ‖iteratedFDerivWithin 𝕜 n (fderivWithin 𝕜 f s) s x‖ =
      ‖iteratedFDerivWithin 𝕜 (n + 1) f s x‖ := by
  -- Porting note: added `comp_apply`.
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : Nat
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ Eq (Norm.norm (iteratedFDerivWithin 𝕜 n (fderivWithin 𝕜 f s) s x)) (Norm.nor …
  -/
  rw [iteratedFDerivWithin_succ_eq_comp_right hs hx, comp_apply, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem iteratedFDerivWithin_one_apply (h : UniqueDiffWithinAt 𝕜 s x) (m : Fin 1 → E) :
    iteratedFDerivWithin 𝕜 1 f s x m = fderivWithin 𝕜 f s x (m 0) := by
  simp only [iteratedFDerivWithin_succ_apply_left, iteratedFDerivWithin_zero_eq_comp,
    (continuousMultilinearCurryFin0 𝕜 E F).symm.comp_fderivWithin h]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    h : UniqueDiffWithinAt 𝕜 s x
    m : Fin 1 → E
    ⊢ Eq ((((↑{ toLinearEquiv := (continuousMultilinearCurryFin0 𝕜 E F).symm.toLin …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- On a set of unique differentiability, the second derivative is obtained by taking the
derivative of the derivative. -/
lemma iteratedFDerivWithin_two_apply (f : E → F) {z : E} (hs : UniqueDiffOn 𝕜 s) (hz : z ∈ s)
    (m : Fin 2 → E) :
    iteratedFDerivWithin 𝕜 2 f s z m = fderivWithin 𝕜 (fderivWithin 𝕜 f s) s z (m 0) (m 1) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    z : E
    hs : UniqueDiffOn 𝕜 s
    hz : Membership.mem s z
    m : Fin 2 → E
    ⊢ Eq ((iteratedFDerivWithin 𝕜 2 f s z) m) (((fderivWithin 𝕜 (fderivWithin 𝕜 f  …
  -/
  simp only [iteratedFDerivWithin_succ_apply_right hs hz]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    z : E
    hs : UniqueDiffOn 𝕜 s
    hz : Membership.mem s z
    m : Fin 2 → E
    ⊢ Eq ((((iteratedFDerivWithin 𝕜 0 (fun y => fderivWithin 𝕜 (fun y => fderivWit …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- On a set of unique differentiability, the second derivative is obtained by taking the
derivative of the derivative. -/
lemma iteratedFDerivWithin_two_apply' (f : E → F) {z : E} (hs : UniqueDiffOn 𝕜 s) (hz : z ∈ s)
    (v w : E) :
    iteratedFDerivWithin 𝕜 2 f s z ![v, w] = fderivWithin 𝕜 (fderivWithin 𝕜 f s) s z v w :=
  iteratedFDerivWithin_two_apply f hs hz _


theorem Filter.EventuallyEq.iteratedFDerivWithin' (h : f₁ =ᶠ[𝓝[s] x] f) (ht : t ⊆ s) (n : ℕ) :
    iteratedFDerivWithin 𝕜 n f₁ t =ᶠ[𝓝[s] x] iteratedFDerivWithin 𝕜 n f t := by
  induction n with
  | zero => exact h.mono fun y hy => DFunLike.ext _ _ fun _ => hy
  | succ n ihn =>
    have : fderivWithin 𝕜 _ t =ᶠ[𝓝[s] x] fderivWithin 𝕜 _ t := ihn.fderivWithin' ht
    refine this.mono fun y hy => ?_
    simp only [iteratedFDerivWithin_succ_eq_comp_left, hy, (· ∘ ·)]


protected theorem Filter.EventuallyEq.iteratedFDerivWithin (h : f₁ =ᶠ[𝓝[s] x] f) (n : ℕ) :
    iteratedFDerivWithin 𝕜 n f₁ s =ᶠ[𝓝[s] x] iteratedFDerivWithin 𝕜 n f s :=
  h.iteratedFDerivWithin' Subset.rfl n


/-- If two functions coincide in a neighborhood of `x` within a set `s` and at `x`, then their
iterated differentials within this set at `x` coincide. -/
theorem Filter.EventuallyEq.iteratedFDerivWithin_eq (h : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x)
    (n : ℕ) : iteratedFDerivWithin 𝕜 n f₁ s x = iteratedFDerivWithin 𝕜 n f s x :=
                                        /-
                                          𝕜 : Type u
                                          inst✝⁴ : NontriviallyNormedField 𝕜
                                          E : Type uE
                                          inst✝³ : NormedAddCommGroup E
                                          inst✝² : NormedSpace 𝕜 E
                                          F : Type uF
                                          inst✝¹ : NormedAddCommGroup F
                                          inst✝ : NormedSpace 𝕜 F
                                          s : Set E
                                          f f₁ : E → F
                                          x : E
                                          h : (nhdsWithin x s).EventuallyEq f₁ f
                                          hx : Eq (f₁ x) (f x)
                                          n : Nat
                                          ⊢ (nhdsWithin x (Insert.insert x s)).EventuallyEq f₁ f
                                        -/
  have : f₁ =ᶠ[𝓝[insert x s] x] f := by simpa [EventuallyEq, hx]
                                        /-
                                          🎉 no goals
                                        -/
  (this.iteratedFDerivWithin' (subset_insert _ _) n).self_of_nhdsWithin (mem_insert _ _)


/-- If two functions coincide on a set `s`, then their iterated differentials within this set
coincide. See also `Filter.EventuallyEq.iteratedFDerivWithin_eq` and
`Filter.EventuallyEq.iteratedFDerivWithin`. -/
theorem iteratedFDerivWithin_congr (hs : EqOn f₁ f s) (hx : x ∈ s) (n : ℕ) :
    iteratedFDerivWithin 𝕜 n f₁ s x = iteratedFDerivWithin 𝕜 n f s x :=
  (hs.eventuallyEq.filter_mono inf_le_right).iteratedFDerivWithin_eq (hs hx) _


/-- If two functions coincide on a set `s`, then their iterated differentials within this set
coincide. See also `Filter.EventuallyEq.iteratedFDerivWithin_eq` and
`Filter.EventuallyEq.iteratedFDerivWithin`. -/
protected theorem Set.EqOn.iteratedFDerivWithin (hs : EqOn f₁ f s) (n : ℕ) :
    EqOn (iteratedFDerivWithin 𝕜 n f₁ s) (iteratedFDerivWithin 𝕜 n f s) s := fun _x hx =>
  iteratedFDerivWithin_congr hs hx n


theorem iteratedFDerivWithin_eventually_congr_set' (y : E) (h : s =ᶠ[𝓝[{y}ᶜ] x] t) (n : ℕ) :
    iteratedFDerivWithin 𝕜 n f s =ᶠ[𝓝 x] iteratedFDerivWithin 𝕜 n f t := by
  induction n generalizing x with
  | zero => rfl
  | succ n ihn =>
    refine (eventually_nhds_nhdsWithin.2 h).mono fun y hy => ?_
    simp only [iteratedFDerivWithin_succ_eq_comp_left, (· ∘ ·)]
    rw [(ihn hy).fderivWithin_eq_nhds, fderivWithin_congr_set' _ hy]


theorem iteratedFDerivWithin_eventually_congr_set (h : s =ᶠ[𝓝 x] t) (n : ℕ) :
    iteratedFDerivWithin 𝕜 n f s =ᶠ[𝓝 x] iteratedFDerivWithin 𝕜 n f t :=
  iteratedFDerivWithin_eventually_congr_set' x (h.filter_mono inf_le_left) n


/-- If two sets coincide in a punctured neighborhood of `x`,
then the corresponding iterated derivatives are equal.

Note that we also allow to puncture the neighborhood of `x` at `y`.
If `y ≠ x`, then this is a no-op. -/
theorem iteratedFDerivWithin_congr_set' {y} (h : s =ᶠ[𝓝[{y}ᶜ] x] t) (n : ℕ) :
    iteratedFDerivWithin 𝕜 n f s x = iteratedFDerivWithin 𝕜 n f t x :=
  (iteratedFDerivWithin_eventually_congr_set' y h n).self_of_nhds


@[simp]
theorem iteratedFDerivWithin_insert {n y} :
    iteratedFDerivWithin 𝕜 n f (insert x s) y = iteratedFDerivWithin 𝕜 n f s y :=
  iteratedFDerivWithin_congr_set' (y := x)
                                          /-
                                            𝕜 : Type u
                                            inst✝⁴ : NontriviallyNormedField 𝕜
                                            E : Type uE
                                            inst✝³ : NormedAddCommGroup E
                                            inst✝² : NormedSpace 𝕜 E
                                            F : Type uF
                                            inst✝¹ : NormedAddCommGroup F
                                            inst✝ : NormedSpace 𝕜 F
                                            s : Set E
                                            f : E → F
                                            x : E
                                            n : Nat
                                            y : E
                                            ⊢ ∀ (x_1 : E), Membership.mem (HasCompl.compl (Singleton.singleton x)) x_1 → I …
                                          -/
    (eventually_mem_nhdsWithin.mono <| by intros; simp_all).set_eq _
                                                  /-
                                                    🎉 no goals
                                                  -/


theorem iteratedFDerivWithin_congr_set (h : s =ᶠ[𝓝 x] t) (n : ℕ) :
    iteratedFDerivWithin 𝕜 n f s x = iteratedFDerivWithin 𝕜 n f t x :=
  (iteratedFDerivWithin_eventually_congr_set h n).self_of_nhds


@[simp]
theorem ftaylorSeriesWithin_insert :
    ftaylorSeriesWithin 𝕜 f (insert x s) = ftaylorSeriesWithin 𝕜 f s := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    ⊢ Eq (ftaylorSeriesWithin 𝕜 f (Insert.insert x s)) (ftaylorSeriesWithin 𝕜 f s)
  -/
  ext y n : 2
  /-
    case h.h
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x y : E
    n : Nat
    ⊢ Eq (ftaylorSeriesWithin 𝕜 f (Insert.insert x s) y n) (ftaylorSeriesWithin 𝕜  …
  -/
  apply iteratedFDerivWithin_insert
  /-
    🎉 no goals
  -/


/-- The iterated differential within a set `s` at a point `x` is not modified if one intersects
`s` with a neighborhood of `x` within `s`. -/
theorem iteratedFDerivWithin_inter' {n : ℕ} (hu : u ∈ 𝓝[s] x) :
    iteratedFDerivWithin 𝕜 n f (s ∩ u) x = iteratedFDerivWithin 𝕜 n f s x :=
  iteratedFDerivWithin_congr_set (nhdsWithin_eq_iff_eventuallyEq.1 <| nhdsWithin_inter_of_mem' hu) _


/-- The iterated differential within a set `s` at a point `x` is not modified if one intersects
`s` with a neighborhood of `x`. -/
theorem iteratedFDerivWithin_inter {n : ℕ} (hu : u ∈ 𝓝 x) :
    iteratedFDerivWithin 𝕜 n f (s ∩ u) x = iteratedFDerivWithin 𝕜 n f s x :=
  iteratedFDerivWithin_inter' (mem_nhdsWithin_of_mem_nhds hu)


/-- The iterated differential within a set `s` at a point `x` is not modified if one intersects
`s` with an open set containing `x`. -/
theorem iteratedFDerivWithin_inter_open {n : ℕ} (hu : IsOpen u) (hx : x ∈ u) :
    iteratedFDerivWithin 𝕜 n f (s ∩ u) x = iteratedFDerivWithin 𝕜 n f s x :=
  iteratedFDerivWithin_inter (hu.mem_nhds hx)


/-- On a set with unique differentiability, any choice of iterated differential has to coincide
with the one we have chosen in `iteratedFDerivWithin 𝕜 m f s`. -/
theorem HasFTaylorSeriesUpToOn.eq_iteratedFDerivWithin_of_uniqueDiffOn
    (h : HasFTaylorSeriesUpToOn n f p s) {m : ℕ} (hmn : m ≤ n) (hs : UniqueDiffOn 𝕜 s)
    (hx : x ∈ s) : p x m = iteratedFDerivWithin 𝕜 m f s x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p s
    m : Nat
    hmn : LE.le (↑m) n
    hs : UniqueDiffOn 𝕜 s
    hx : Membership.mem s x
    ⊢ Eq (p x m) (iteratedFDerivWithin 𝕜 m f s x)
  -/
  induction' m with m IH generalizing x
    /-
      case zero
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      n : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      h : HasFTaylorSeriesUpToOn n f p s
      hs : UniqueDiffOn 𝕜 s
      x : E
      hmn : LE.le (↑0) n
      hx : Membership.mem s x
      ⊢ Eq (p x 0) (iteratedFDerivWithin 𝕜 0 f s x)
    -/
  · rw [h.zero_eq' hx, iteratedFDerivWithin_zero_eq_comp]; rfl
                                                           /-
                                                             🎉 no goals
                                                           -/
    /-
      case succ
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      n : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      h : HasFTaylorSeriesUpToOn n f p s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      IH : ∀ {x : E}, LE.le (↑m) n → Membership.mem s x → Eq (p x m) (iteratedFDeriv …
      x : E
      hmn : LE.le (↑(HAdd.hAdd m 1)) n
      hx : Membership.mem s x
      ⊢ Eq (p x (HAdd.hAdd m 1)) (iteratedFDerivWithin 𝕜 (HAdd.hAdd m 1) f s x)
    -/
  · have A : m < n := lt_of_lt_of_le (mod_cast lt_add_one m) hmn
    have :
      HasFDerivWithinAt (fun y : E => iteratedFDerivWithin 𝕜 m f s y)
        (ContinuousMultilinearMap.curryLeft (p x (Nat.succ m))) s x :=
      (h.fderivWithin m A x hx).congr (fun y hy => (IH (le_of_lt A) hy).symm)
        (IH (le_of_lt A) hx).symm
    /-
      case succ
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      n : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      h : HasFTaylorSeriesUpToOn n f p s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      IH : ∀ {x : E}, LE.le (↑m) n → Membership.mem s x → Eq (p x m) (iteratedFDeriv …
      x : E
      hmn : LE.le (↑(HAdd.hAdd m 1)) n
      hx : Membership.mem s x
      A : LT.lt (↑m) n
      this : HasFDerivWithinAt (fun y => iteratedFDerivWithin 𝕜 m f s y) (p x m.succ …
      ⊢ Eq (p x (HAdd.hAdd m 1)) (iteratedFDerivWithin 𝕜 (HAdd.hAdd m 1) f s x)
    -/
    rw [iteratedFDerivWithin_succ_eq_comp_left, Function.comp_apply, this.fderivWithin (hs x hx)]
    /-
      case succ
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      s : Set E
      f : E → F
      n : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      h : HasFTaylorSeriesUpToOn n f p s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      IH : ∀ {x : E}, LE.le (↑m) n → Membership.mem s x → Eq (p x m) (iteratedFDeriv …
      x : E
      hmn : LE.le (↑(HAdd.hAdd m 1)) n
      hx : Membership.mem s x
      A : LT.lt (↑m) n
      this : HasFDerivWithinAt (fun y => iteratedFDerivWithin 𝕜 m f s y) (p x m.succ …
      ⊢ Eq (p x (HAdd.hAdd m 1)) ((continuousMultilinearCurryLeftEquiv 𝕜 (fun x => E …
    -/
    exact (ContinuousMultilinearMap.uncurry_curryLeft _).symm
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-03-28")]
alias HasFTaylorSeriesUpToOn.eq_ftaylor_series_of_uniqueDiffOn :=
  HasFTaylorSeriesUpToOn.eq_iteratedFDerivWithin_of_uniqueDiffOn


/-- The iterated derivative commutes with shifting the function by a constant on the left. -/
lemma iteratedFDerivWithin_comp_add_left' (n : ℕ) (a : E) :
    iteratedFDerivWithin 𝕜 n (fun z ↦ f (a + z)) s =
      fun x ↦ iteratedFDerivWithin 𝕜 n f (a +ᵥ s) (a + x) := by
  induction n with
  | zero => simp [iteratedFDerivWithin]
  | succ n IH =>
    ext v
    rw [iteratedFDerivWithin_succ_eq_comp_left, iteratedFDerivWithin_succ_eq_comp_left]
    simp only [Nat.succ_eq_add_one, IH, comp_apply, continuousMultilinearCurryLeftEquiv_symm_apply]
    congr 2
    rw [fderivWithin_comp_add_left]


/-- The iterated derivative commutes with shifting the function by a constant on the left. -/
lemma iteratedFDerivWithin_comp_add_left (n : ℕ) (a : E) (x : E) :
    iteratedFDerivWithin 𝕜 n (fun z ↦ f (a + z)) s x =
      iteratedFDerivWithin 𝕜 n f (a +ᵥ s) (a + x) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    n : Nat
    a x : E
    ⊢ Eq (iteratedFDerivWithin 𝕜 n (fun z => f (HAdd.hAdd a z)) s x) (iteratedFDer …
  -/
  simp [iteratedFDerivWithin_comp_add_left']
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with shifting the function by a constant on the right. -/
lemma iteratedFDerivWithin_comp_add_right' (n : ℕ) (a : E) :
    iteratedFDerivWithin 𝕜 n (fun z ↦ f (z + a)) s =
      fun x ↦ iteratedFDerivWithin 𝕜 n f (a +ᵥ s) (x + a) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    n : Nat
    a : E
    ⊢ Eq (iteratedFDerivWithin 𝕜 n (fun z => f (HAdd.hAdd z a)) s) fun x => iterat …
  -/
  simpa [add_comm a] using iteratedFDerivWithin_comp_add_left' n a
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with shifting the function by a constant on the right. -/
lemma iteratedFDerivWithin_comp_add_right (n : ℕ) (a : E) (x : E) :
    iteratedFDerivWithin 𝕜 n (fun z ↦ f (z + a)) s x =
      iteratedFDerivWithin 𝕜 n f (a +ᵥ s) (x + a) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    n : Nat
    a x : E
    ⊢ Eq (iteratedFDerivWithin 𝕜 n (fun z => f (HAdd.hAdd z a)) s x) (iteratedFDer …
  -/
  simp [iteratedFDerivWithin_comp_add_right']
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with subtracting a constant. -/
lemma iteratedFDerivWithin_comp_sub' (n : ℕ) (a : E) :
    iteratedFDerivWithin 𝕜 n (fun z ↦ f (z - a)) s =
      fun x ↦ iteratedFDerivWithin 𝕜 n f (-a +ᵥ s) (x - a) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    n : Nat
    a : E
    ⊢ Eq (iteratedFDerivWithin 𝕜 n (fun z => f (HSub.hSub z a)) s) fun x => iterat …
  -/
  simpa [sub_eq_add_neg] using iteratedFDerivWithin_comp_add_right' n (-a)
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with subtracting a constant. -/
lemma iteratedFDerivWithin_comp_sub (n : ℕ) (a : E) :
    iteratedFDerivWithin 𝕜 n (fun z ↦ f (z - a)) s x =
      iteratedFDerivWithin 𝕜 n f (-a +ᵥ s) (x - a) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    n : Nat
    a : E
    ⊢ Eq (iteratedFDerivWithin 𝕜 n (fun z => f (HSub.hSub z a)) s x) (iteratedFDer …
  -/
  simp [iteratedFDerivWithin_comp_sub']
  /-
    🎉 no goals
  -/


/-- `HasFTaylorSeriesUpTo n f p` registers the fact that `p 0 = f` and `p (m+1)` is a
derivative of `p m` for `m < n`, and is continuous for `m ≤ n`. This is a predicate analogous to
`HasFDerivAt` but for higher order derivatives.

Notice that `p` does not sum up to `f` on the diagonal (`FormalMultilinearSeries.sum`), even if
`f` is analytic and `n = ∞`: an addition `1/m!` factor on the `m`th term is necessary for that. -/
structure HasFTaylorSeriesUpTo
  (n : WithTop ℕ∞) (f : E → F) (p : E → FormalMultilinearSeries 𝕜 E F) : Prop where
  zero_eq : ∀ x, (p x 0).curry0 = f x
  fderiv : ∀ m : ℕ, m < n → ∀ x, HasFDerivAt (fun y => p y m) (p x m.succ).curryLeft x
  cont : ∀ m : ℕ, m ≤ n → Continuous fun x => p x m


theorem HasFTaylorSeriesUpTo.zero_eq' (h : HasFTaylorSeriesUpTo n f p) (x : E) :
    p x 0 = (continuousMultilinearCurryFin0 𝕜 E F).symm (f x) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpTo n f p
    x : E
    ⊢ Eq (p x 0) ((continuousMultilinearCurryFin0 𝕜 E F).symm (f x))
  -/
  rw [← h.zero_eq x]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpTo n f p
    x : E
    ⊢ Eq (p x 0) ((continuousMultilinearCurryFin0 𝕜 E F).symm (p x 0).curry0)
  -/
  exact (p x 0).uncurry0_curry0.symm
  /-
    🎉 no goals
  -/


theorem hasFTaylorSeriesUpToOn_univ_iff :
    HasFTaylorSeriesUpToOn n f p univ ↔ HasFTaylorSeriesUpTo n f p := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    ⊢ Iff (HasFTaylorSeriesUpToOn n f p Set.univ) (HasFTaylorSeriesUpTo n f p)
  -/
  constructor
    /-
      case mp
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      n : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      ⊢ HasFTaylorSeriesUpToOn n f p Set.univ → HasFTaylorSeriesUpTo n f p
    -/
  · intro H
    /-
      case mp
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      n : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      H : HasFTaylorSeriesUpToOn n f p Set.univ
      ⊢ HasFTaylorSeriesUpTo n f p
    -/
    constructor
      /-
        case mp.zero_eq
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpToOn n f p Set.univ
        ⊢ ∀ (x : E), Eq (p x 0).curry0 (f x)
      -/
    · exact fun x => H.zero_eq x (mem_univ x)
      /-
        🎉 no goals
      -/
      /-
        case mp.fderiv
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpToOn n f p Set.univ
        ⊢ ∀ (m : Nat), LT.lt (↑m) n → ∀ (x : E), HasFDerivAt (fun y => p y m) (p x m.s …
      -/
    · intro m hm x
      /-
        case mp.fderiv
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpToOn n f p Set.univ
        m : Nat
        hm : LT.lt (↑m) n
        x : E
        ⊢ HasFDerivAt (fun y => p y m) (p x m.succ).curryLeft x
      -/
      rw [← hasFDerivWithinAt_univ]
      /-
        case mp.fderiv
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpToOn n f p Set.univ
        m : Nat
        hm : LT.lt (↑m) n
        x : E
        ⊢ HasFDerivWithinAt (fun y => p y m) (p x m.succ).curryLeft Set.univ x
      -/
      exact H.fderivWithin m hm x (mem_univ x)
      /-
        🎉 no goals
      -/
      /-
        case mp.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpToOn n f p Set.univ
        ⊢ ∀ (m : Nat), LE.le (↑m) n → Continuous fun x => p x m
      -/
    · intro m hm
      /-
        case mp.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpToOn n f p Set.univ
        m : Nat
        hm : LE.le (↑m) n
        ⊢ Continuous fun x => p x m
      -/
      rw [continuous_iff_continuousOn_univ]
      /-
        case mp.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpToOn n f p Set.univ
        m : Nat
        hm : LE.le (↑m) n
        ⊢ ContinuousOn (fun x => p x m) Set.univ
      -/
      exact H.cont m hm
      /-
        🎉 no goals
      -/
    /-
      case mpr
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      n : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      ⊢ HasFTaylorSeriesUpTo n f p → HasFTaylorSeriesUpToOn n f p Set.univ
    -/
  · intro H
    /-
      case mpr
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      n : WithTop ENat
      p : E → FormalMultilinearSeries 𝕜 E F
      H : HasFTaylorSeriesUpTo n f p
      ⊢ HasFTaylorSeriesUpToOn n f p Set.univ
    -/
    constructor
      /-
        case mpr.zero_eq
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpTo n f p
        ⊢ ∀ (x : E), Membership.mem Set.univ x → Eq (p x 0).curry0 (f x)
      -/
    · exact fun x _ => H.zero_eq x
      /-
        🎉 no goals
      -/
      /-
        case mpr.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpTo n f p
        ⊢ ∀ (m : Nat), LT.lt (↑m) n → ∀ (x : E), Membership.mem Set.univ x → HasFDeriv …
      -/
    · intro m hm x _
      /-
        case mpr.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpTo n f p
        m : Nat
        hm : LT.lt (↑m) n
        x : E
        a✝ : Membership.mem Set.univ x
        ⊢ HasFDerivWithinAt (fun x => p x m) (p x m.succ).curryLeft Set.univ x
      -/
      rw [hasFDerivWithinAt_univ]
      /-
        case mpr.fderivWithin
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpTo n f p
        m : Nat
        hm : LT.lt (↑m) n
        x : E
        a✝ : Membership.mem Set.univ x
        ⊢ HasFDerivAt (fun x => p x m) (p x m.succ).curryLeft x
      -/
      exact H.fderiv m hm x
      /-
        🎉 no goals
      -/
      /-
        case mpr.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpTo n f p
        ⊢ ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => p x m) Set.univ
      -/
    · intro m hm
      /-
        case mpr.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpTo n f p
        m : Nat
        hm : LE.le (↑m) n
        ⊢ ContinuousOn (fun x => p x m) Set.univ
      -/
      rw [← continuous_iff_continuousOn_univ]
      /-
        case mpr.cont
        𝕜 : Type u
        inst✝⁴ : NontriviallyNormedField 𝕜
        E : Type uE
        inst✝³ : NormedAddCommGroup E
        inst✝² : NormedSpace 𝕜 E
        F : Type uF
        inst✝¹ : NormedAddCommGroup F
        inst✝ : NormedSpace 𝕜 F
        f : E → F
        n : WithTop ENat
        p : E → FormalMultilinearSeries 𝕜 E F
        H : HasFTaylorSeriesUpTo n f p
        m : Nat
        hm : LE.le (↑m) n
        ⊢ Continuous fun x => p x m
      -/
      exact H.cont m hm
      /-
        🎉 no goals
      -/


theorem HasFTaylorSeriesUpTo.hasFTaylorSeriesUpToOn (h : HasFTaylorSeriesUpTo n f p) (s : Set E) :
    HasFTaylorSeriesUpToOn n f p s :=
  (hasFTaylorSeriesUpToOn_univ_iff.2 h).mono (subset_univ _)


theorem HasFTaylorSeriesUpTo.of_le (h : HasFTaylorSeriesUpTo n f p) (hmn : m ≤ n) :
    HasFTaylorSeriesUpTo m f p := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    m n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpTo n f p
    hmn : LE.le m n
    ⊢ HasFTaylorSeriesUpTo m f p
  -/
  rw [← hasFTaylorSeriesUpToOn_univ_iff] at h ⊢; exact h.of_le hmn
                                                 /-
                                                   🎉 no goals
                                                 -/


@[deprecated (since := "2024-11-07")]
alias HasFTaylorSeriesUpTo.ofLe := HasFTaylorSeriesUpTo.of_le


theorem HasFTaylorSeriesUpTo.continuous (h : HasFTaylorSeriesUpTo n f p) : Continuous f := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpTo n f p
    ⊢ Continuous f
  -/
  rw [← hasFTaylorSeriesUpToOn_univ_iff] at h
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p Set.univ
    ⊢ Continuous f
  -/
  rw [continuous_iff_continuousOn_univ]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p Set.univ
    ⊢ ContinuousOn f Set.univ
  -/
  exact h.continuousOn
  /-
    🎉 no goals
  -/


theorem hasFTaylorSeriesUpTo_zero_iff :
    HasFTaylorSeriesUpTo 0 f p ↔ Continuous f ∧ ∀ x, (p x 0).curry0 = f x := by
  simp [hasFTaylorSeriesUpToOn_univ_iff.symm, continuous_iff_continuousOn_univ,
    hasFTaylorSeriesUpToOn_zero_iff]


theorem hasFTaylorSeriesUpTo_top_iff (hN : ∞ ≤ N) :
    HasFTaylorSeriesUpTo N f p ↔ ∀ n : ℕ, HasFTaylorSeriesUpTo n f p := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    N : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    hN : LE.le (↑Top.top) N
    ⊢ Iff (HasFTaylorSeriesUpTo N f p) (∀ (n : Nat), HasFTaylorSeriesUpTo (↑n) f p)
  -/
  simp only [← hasFTaylorSeriesUpToOn_univ_iff, hasFTaylorSeriesUpToOn_top_iff hN]
  /-
    🎉 no goals
  -/


/-- In the case that `n = ∞` we don't need the continuity assumption in
`HasFTaylorSeriesUpTo`. -/
theorem hasFTaylorSeriesUpTo_top_iff' (hN : ∞ ≤ N) :
    HasFTaylorSeriesUpTo N f p ↔
      (∀ x, (p x 0).curry0 = f x) ∧
        ∀ (m : ℕ) (x), HasFDerivAt (fun y => p y m) (p x m.succ).curryLeft x := by
  simp only [← hasFTaylorSeriesUpToOn_univ_iff, hasFTaylorSeriesUpToOn_top_iff' hN, mem_univ,
    forall_true_left, hasFDerivWithinAt_univ]


/-- If a function has a Taylor series at order at least `1`, then the term of order `1` of this
series is a derivative of `f`. -/
theorem HasFTaylorSeriesUpTo.hasFDerivAt (h : HasFTaylorSeriesUpTo n f p) (hn : 1 ≤ n) (x : E) :
    HasFDerivAt f (continuousMultilinearCurryFin1 𝕜 E F (p x 1)) x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpTo n f p
    hn : LE.le 1 n
    x : E
    ⊢ HasFDerivAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p x 1)) x
  -/
  rw [← hasFDerivWithinAt_univ]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpTo n f p
    hn : LE.le 1 n
    x : E
    ⊢ HasFDerivWithinAt f ((continuousMultilinearCurryFin1 𝕜 E F) (p x 1)) Set.uni …
  -/
  exact (hasFTaylorSeriesUpToOn_univ_iff.2 h).hasFDerivWithinAt hn (mem_univ _)
  /-
    🎉 no goals
  -/


theorem HasFTaylorSeriesUpTo.differentiable (h : HasFTaylorSeriesUpTo n f p) (hn : 1 ≤ n) :
    Differentiable 𝕜 f := fun x => (h.hasFDerivAt hn x).differentiableAt


/-- `p` is a Taylor series of `f` up to `n+1` if and only if `p.shift` is a Taylor series up to `n`
for `p 1`, which is a derivative of `f`. -/
theorem hasFTaylorSeriesUpTo_succ_nat_iff_right {n : ℕ} :
    HasFTaylorSeriesUpTo (n + 1 : ℕ) f p ↔
      (∀ x, (p x 0).curry0 = f x) ∧
        (∀ x, HasFDerivAt (fun y => p y 0) (p x 1).curryLeft x) ∧
          HasFTaylorSeriesUpTo n (fun x => continuousMultilinearCurryFin1 𝕜 E F (p x 1)) fun x =>
            (p x).shift := by
  simp only [hasFTaylorSeriesUpToOn_succ_nat_iff_right, ← hasFTaylorSeriesUpToOn_univ_iff, mem_univ,
    forall_true_left, hasFDerivWithinAt_univ]


@[deprecated (since := "2024-11-07")]
alias hasFTaylorSeriesUpTo_succ_iff_right := hasFTaylorSeriesUpTo_succ_nat_iff_right


/-- The `n`-th derivative of a function, as a multilinear map, defined inductively. -/
noncomputable def iteratedFDeriv (n : ℕ) (f : E → F) : E → E[×n]→L[𝕜] F :=
  Nat.recOn n (fun x => ContinuousMultilinearMap.uncurry0 𝕜 E (f x)) fun _ rec x =>
    ContinuousLinearMap.uncurryLeft (fderiv 𝕜 rec x)


/-- Formal Taylor series associated to a function. -/
def ftaylorSeries (f : E → F) (x : E) : FormalMultilinearSeries 𝕜 E F := fun n =>
  iteratedFDeriv 𝕜 n f x


@[simp]
theorem iteratedFDeriv_zero_apply (m : Fin 0 → E) :
    (iteratedFDeriv 𝕜 0 f x : (Fin 0 → E) → F) m = f x :=
  rfl


theorem iteratedFDeriv_zero_eq_comp :
    iteratedFDeriv 𝕜 0 f = (continuousMultilinearCurryFin0 𝕜 E F).symm ∘ f :=
  rfl


@[simp]
theorem norm_iteratedFDeriv_zero : ‖iteratedFDeriv 𝕜 0 f x‖ = ‖f x‖ := by
  -- Porting note: added `comp_apply`.
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    ⊢ Eq (Norm.norm (iteratedFDeriv 𝕜 0 f x)) (Norm.norm (f x))
  -/
  rw [iteratedFDeriv_zero_eq_comp, comp_apply, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


theorem iteratedFDerivWithin_zero_eq : iteratedFDerivWithin 𝕜 0 f s = iteratedFDeriv 𝕜 0 f := rfl


theorem iteratedFDeriv_succ_apply_left {n : ℕ} (m : Fin (n + 1) → E) :
    (iteratedFDeriv 𝕜 (n + 1) f x : (Fin (n + 1) → E) → F) m =
      (fderiv 𝕜 (iteratedFDeriv 𝕜 n f) x : E → E[×n]→L[𝕜] F) (m 0) (tail m) :=
  rfl


/-- Writing explicitly the `n+1`-th derivative as the composition of a currying linear equiv,
and the derivative of the `n`-th derivative. -/
theorem iteratedFDeriv_succ_eq_comp_left {n : ℕ} :
    iteratedFDeriv 𝕜 (n + 1) f =
      (continuousMultilinearCurryLeftEquiv 𝕜 (fun _ : Fin (n + 1) => E) F).symm ∘
        fderiv 𝕜 (iteratedFDeriv 𝕜 n f) :=
  rfl


/-- Writing explicitly the derivative of the `n`-th derivative as the composition of a currying
linear equiv, and the `n + 1`-th derivative. -/
theorem fderiv_iteratedFDeriv {n : ℕ} :
    fderiv 𝕜 (iteratedFDeriv 𝕜 n f) =
      continuousMultilinearCurryLeftEquiv 𝕜 (fun _ : Fin (n + 1) => E) F ∘
        iteratedFDeriv 𝕜 (n + 1) f :=
  rfl


theorem tsupport_iteratedFDeriv_subset (n : ℕ) : tsupport (iteratedFDeriv 𝕜 n f) ⊆ tsupport f := by
  induction n with
  | zero =>
    rw [iteratedFDeriv_zero_eq_comp]
    exact closure_minimal ((support_comp_subset (LinearIsometryEquiv.map_zero _) _).trans
      subset_closure) isClosed_closure
  | succ n IH =>
    rw [iteratedFDeriv_succ_eq_comp_left]
    exact closure_minimal ((support_comp_subset (LinearIsometryEquiv.map_zero _) _).trans
      ((support_fderiv_subset 𝕜).trans IH)) isClosed_closure


theorem support_iteratedFDeriv_subset (n : ℕ) : support (iteratedFDeriv 𝕜 n f) ⊆ tsupport f :=
  subset_closure.trans (tsupport_iteratedFDeriv_subset n)


theorem HasCompactSupport.iteratedFDeriv (hf : HasCompactSupport f) (n : ℕ) :
    HasCompactSupport (iteratedFDeriv 𝕜 n f) :=
  hf.of_isClosed_subset isClosed_closure (tsupport_iteratedFDeriv_subset n)


theorem norm_fderiv_iteratedFDeriv {n : ℕ} :
    ‖fderiv 𝕜 (iteratedFDeriv 𝕜 n f) x‖ = ‖iteratedFDeriv 𝕜 (n + 1) f x‖ := by
  -- Porting note: added `comp_apply`.
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : Nat
    ⊢ Eq (Norm.norm (fderiv 𝕜 (iteratedFDeriv 𝕜 n f) x)) (Norm.norm (iteratedFDeri …
  -/
  rw [iteratedFDeriv_succ_eq_comp_left, comp_apply, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


theorem iteratedFDerivWithin_univ {n : ℕ} :
    iteratedFDerivWithin 𝕜 n f univ = iteratedFDeriv 𝕜 n f := by
  induction n with
  | zero => ext x; simp
  | succ n IH =>
    ext x m
    rw [iteratedFDeriv_succ_apply_left, iteratedFDerivWithin_succ_apply_left, IH, fderivWithin_univ]


theorem HasFTaylorSeriesUpTo.eq_iteratedFDeriv
    (h : HasFTaylorSeriesUpTo n f p) {m : ℕ} (hmn : m ≤ n) (x : E) :
    p x m = iteratedFDeriv 𝕜 m f x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpTo n f p
    m : Nat
    hmn : LE.le (↑m) n
    x : E
    ⊢ Eq (p x m) (iteratedFDeriv 𝕜 m f x)
  -/
  rw [← iteratedFDerivWithin_univ]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpTo n f p
    m : Nat
    hmn : LE.le (↑m) n
    x : E
    ⊢ Eq (p x m) (iteratedFDerivWithin 𝕜 m f Set.univ x)
  -/
  rw [← hasFTaylorSeriesUpToOn_univ_iff] at h
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : WithTop ENat
    p : E → FormalMultilinearSeries 𝕜 E F
    h : HasFTaylorSeriesUpToOn n f p Set.univ
    m : Nat
    hmn : LE.le (↑m) n
    x : E
    ⊢ Eq (p x m) (iteratedFDerivWithin 𝕜 m f Set.univ x)
  -/
  exact h.eq_iteratedFDerivWithin_of_uniqueDiffOn hmn uniqueDiffOn_univ (mem_univ _)
  /-
    🎉 no goals
  -/


/-- In an open set, the iterated derivative within this set coincides with the global iterated
derivative. -/
theorem iteratedFDerivWithin_of_isOpen (n : ℕ) (hs : IsOpen s) :
    EqOn (iteratedFDerivWithin 𝕜 n f s) (iteratedFDeriv 𝕜 n f) s := by
  induction n with
  | zero =>
    intro x _
    ext1
    simp only [iteratedFDerivWithin_zero_apply, iteratedFDeriv_zero_apply]
  | succ n IH =>
    intro x hx
    rw [iteratedFDeriv_succ_eq_comp_left, iteratedFDerivWithin_succ_eq_comp_left]
    dsimp
    congr 1
    rw [fderivWithin_of_isOpen hs hx]
    apply Filter.EventuallyEq.fderiv_eq
    filter_upwards [hs.mem_nhds hx]
    exact IH


theorem ftaylorSeriesWithin_univ : ftaylorSeriesWithin 𝕜 f univ = ftaylorSeries 𝕜 f := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    ⊢ Eq (ftaylorSeriesWithin 𝕜 f Set.univ) (ftaylorSeries 𝕜 f)
  -/
  ext1 x; ext1 n
  /-
    case h.h
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : Nat
    ⊢ Eq (ftaylorSeriesWithin 𝕜 f Set.univ x n) (ftaylorSeries 𝕜 f x n)
  -/
  change iteratedFDerivWithin 𝕜 n f univ x = iteratedFDeriv 𝕜 n f x
  /-
    case h.h
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : Nat
    ⊢ Eq (iteratedFDerivWithin 𝕜 n f Set.univ x) (iteratedFDeriv 𝕜 n f x)
  -/
  rw [iteratedFDerivWithin_univ]
  /-
    🎉 no goals
  -/


theorem iteratedFDeriv_succ_apply_right {n : ℕ} (m : Fin (n + 1) → E) :
    (iteratedFDeriv 𝕜 (n + 1) f x : (Fin (n + 1) → E) → F) m =
      iteratedFDeriv 𝕜 n (fun y => fderiv 𝕜 f y) x (init m) (m (last n)) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : Nat
    m : Fin (HAdd.hAdd n 1) → E
    ⊢ Eq ((iteratedFDeriv 𝕜 (HAdd.hAdd n 1) f x) m) (((iteratedFDeriv 𝕜 n (fun y = …
  -/
  rw [← iteratedFDerivWithin_univ, ← iteratedFDerivWithin_univ, ← fderivWithin_univ]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : Nat
    m : Fin (HAdd.hAdd n 1) → E
    ⊢ Eq ((iteratedFDerivWithin 𝕜 (HAdd.hAdd n 1) f Set.univ x) m) (((iteratedFDer …
  -/
  exact iteratedFDerivWithin_succ_apply_right uniqueDiffOn_univ (mem_univ _) _
  /-
    🎉 no goals
  -/


/-- Writing explicitly the `n+1`-th derivative as the composition of a currying linear equiv,
and the `n`-th derivative of the derivative. -/
theorem iteratedFDeriv_succ_eq_comp_right {n : ℕ} :
    iteratedFDeriv 𝕜 (n + 1) f x =
      ((continuousMultilinearCurryRightEquiv' 𝕜 n E F).symm ∘
          iteratedFDeriv 𝕜 n fun y => fderiv 𝕜 f y) x := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : Nat
    ⊢ Eq (iteratedFDeriv 𝕜 (HAdd.hAdd n 1) f x) (Function.comp (⇑(continuousMultil …
  -/
  ext m; rw [iteratedFDeriv_succ_apply_right]; rfl
                                               /-
                                                 🎉 no goals
                                               -/


theorem norm_iteratedFDeriv_fderiv {n : ℕ} :
    ‖iteratedFDeriv 𝕜 n (fderiv 𝕜 f) x‖ = ‖iteratedFDeriv 𝕜 (n + 1) f x‖ := by
  -- Porting note: added `comp_apply`.
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : Nat
    ⊢ Eq (Norm.norm (iteratedFDeriv 𝕜 n (fderiv 𝕜 f) x)) (Norm.norm (iteratedFDeri …
  -/
  rw [iteratedFDeriv_succ_eq_comp_right, comp_apply, LinearIsometryEquiv.norm_map]
  /-
    🎉 no goals
  -/


@[simp]
theorem iteratedFDeriv_one_apply (m : Fin 1 → E) :
    iteratedFDeriv 𝕜 1 f x m = fderiv 𝕜 f x (m 0) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    x : E
    m : Fin 1 → E
    ⊢ Eq ((iteratedFDeriv 𝕜 1 f x) m) ((fderiv 𝕜 f x) (m 0))
  -/
  rw [iteratedFDeriv_succ_apply_right, iteratedFDeriv_zero_apply]; rfl
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


lemma iteratedFDeriv_two_apply (f : E → F) (z : E) (m : Fin 2 → E) :
    iteratedFDeriv 𝕜 2 f z m = fderiv 𝕜 (fderiv 𝕜 f) z (m 0) (m 1) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    z : E
    m : Fin 2 → E
    ⊢ Eq ((iteratedFDeriv 𝕜 2 f z) m) (((fderiv 𝕜 (fderiv 𝕜 f) z) (m 0)) (m 1))
  -/
  simp only [iteratedFDeriv_succ_apply_right]
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    z : E
    m : Fin 2 → E
    ⊢ Eq ((((iteratedFDeriv 𝕜 0 (fun y => fderiv 𝕜 (fun y => fderiv 𝕜 f y) y) z) ( …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with shifting the function by a constant on the left. -/
lemma iteratedFDeriv_comp_add_left' (n : ℕ) (a : E) :
    iteratedFDeriv 𝕜 n (fun z ↦ f (a + z)) = fun x ↦ iteratedFDeriv 𝕜 n f (a + x) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : Nat
    a : E
    ⊢ Eq (iteratedFDeriv 𝕜 n fun z => f (HAdd.hAdd a z)) fun x => iteratedFDeriv 𝕜 …
  -/
  simpa [← iteratedFDerivWithin_univ] using iteratedFDerivWithin_comp_add_left' n a (s := univ)
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with shifting the function by a constant on the left. -/
lemma iteratedFDeriv_comp_add_left (n : ℕ) (a : E) (x : E) :
    iteratedFDeriv 𝕜 n (fun z ↦ f (a + z)) x = iteratedFDeriv 𝕜 n f (a + x) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : Nat
    a x : E
    ⊢ Eq (iteratedFDeriv 𝕜 n (fun z => f (HAdd.hAdd a z)) x) (iteratedFDeriv 𝕜 n f …
  -/
  simp [iteratedFDeriv_comp_add_left']
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with shifting the function by a constant on the right. -/
lemma iteratedFDeriv_comp_add_right' (n : ℕ) (a : E) :
    iteratedFDeriv 𝕜 n (fun z ↦ f (z + a)) = fun x ↦ iteratedFDeriv 𝕜 n f (x + a) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : Nat
    a : E
    ⊢ Eq (iteratedFDeriv 𝕜 n fun z => f (HAdd.hAdd z a)) fun x => iteratedFDeriv 𝕜 …
  -/
  simpa [add_comm a] using iteratedFDeriv_comp_add_left' n a
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with shifting the function by a constant on the right. -/
lemma iteratedFDeriv_comp_add_right (n : ℕ) (a : E) (x : E) :
    iteratedFDeriv 𝕜 n (fun z ↦ f (z + a)) x = iteratedFDeriv 𝕜 n f (x + a) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : Nat
    a x : E
    ⊢ Eq (iteratedFDeriv 𝕜 n (fun z => f (HAdd.hAdd z a)) x) (iteratedFDeriv 𝕜 n f …
  -/
  simp [iteratedFDeriv_comp_add_right']
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with subtracting a constant. -/
lemma iteratedFDeriv_comp_sub' (n : ℕ) (a : E) :
    iteratedFDeriv 𝕜 n (fun z ↦ f (z - a)) = fun x ↦ iteratedFDeriv 𝕜 n f (x - a) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : Nat
    a : E
    ⊢ Eq (iteratedFDeriv 𝕜 n fun z => f (HSub.hSub z a)) fun x => iteratedFDeriv 𝕜 …
  -/
  simpa [sub_eq_add_neg] using iteratedFDeriv_comp_add_right' n (-a)
  /-
    🎉 no goals
  -/


/-- The iterated derivative commutes with subtracting a constant. -/
lemma iteratedFDeriv_comp_sub (n : ℕ) (a : E) (x : E) :
    iteratedFDeriv 𝕜 n (fun z ↦ f (z - a)) x = iteratedFDeriv 𝕜 n f (x - a) := by
  /-
    𝕜 : Type u
    inst✝⁴ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝³ : NormedAddCommGroup E
    inst✝² : NormedSpace 𝕜 E
    F : Type uF
    inst✝¹ : NormedAddCommGroup F
    inst✝ : NormedSpace 𝕜 F
    f : E → F
    n : Nat
    a x : E
    ⊢ Eq (iteratedFDeriv 𝕜 n (fun z => f (HSub.hSub z a)) x) (iteratedFDeriv 𝕜 n f …
  -/
  simp [iteratedFDeriv_comp_sub']
  /-
    🎉 no goals
  -/

