variable (𝕜) in
/-- A function is continuously differentiable up to order `n` within a set `s` at a point `x` if
it admits continuous derivatives up to order `n` in a neighborhood of `x` in `s ∪ {x}`.
For `n = ∞`, we only require that this holds up to any finite order (where the neighborhood may
depend on the finite order we consider).
For `n = ω`, we require the function to be analytic within `s` at `x`. The precise definition we
give (all the derivatives should be analytic) is more involved to work around issues when the space
is not complete, but it is equivalent when the space is complete.

For instance, a real function which is `C^m` on `(-1/m, 1/m)` for each natural `m`, but not
better, is `C^∞` at `0` within `univ`.
-/
def ContDiffWithinAt (n : WithTop ℕ∞) (f : E → F) (s : Set E) (x : E) : Prop :=
  match n with
  | ω => ∃ u ∈ 𝓝[insert x s] x, ∃ p : E → FormalMultilinearSeries 𝕜 E F,
      HasFTaylorSeriesUpToOn ω f p u ∧ ∀ i, AnalyticOn 𝕜 (fun x ↦ p x i) u
  | (n : ℕ∞) => ∀ m : ℕ, m ≤ n → ∃ u ∈ 𝓝[insert x s] x,
      ∃ p : E → FormalMultilinearSeries 𝕜 E F, HasFTaylorSeriesUpToOn m f p u


lemma HasFTaylorSeriesUpToOn.analyticOn
    (hf : HasFTaylorSeriesUpToOn ω f p s) (h : AnalyticOn 𝕜 (fun x ↦ p x 0) s) :
    AnalyticOn 𝕜 f s := by
  have : AnalyticOn 𝕜 (fun x ↦ (continuousMultilinearCurryFin0 𝕜 E F) (p x 0)) s :=
    (LinearIsometryEquiv.analyticOnNhd _ _ ).comp_analyticOn
      h (Set.mapsTo_univ _ _)
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
    hf : HasFTaylorSeriesUpToOn Top.top f p s
    h : AnalyticOn 𝕜 (fun x => p x 0) s
    this : AnalyticOn 𝕜 (fun x => (continuousMultilinearCurryFin0 𝕜 E F) (p x 0)) s
    ⊢ AnalyticOn 𝕜 f s
  -/
  exact this.congr (fun y hy ↦ (hf.zero_eq _ hy).symm)
  /-
    🎉 no goals
  -/


lemma ContDiffWithinAt.analyticOn (h : ContDiffWithinAt 𝕜 ω f s x) :
    ∃ u ∈ 𝓝[insert x s] x, AnalyticOn 𝕜 f u := by
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
    h : ContDiffWithinAt 𝕜 Top.top f s x
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
  -/
  obtain ⟨u, hu, p, hp, h'p⟩ := h
  /-
    case intro.intro.intro.intro
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
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    p : E → FormalMultilinearSeries 𝕜 E F
    hp : HasFTaylorSeriesUpToOn Top.top f p u
    h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
  -/
  exact ⟨u, hu, hp.analyticOn (h'p 0)⟩
  /-
    🎉 no goals
  -/


lemma ContDiffWithinAt.analyticWithinAt (h : ContDiffWithinAt 𝕜 ω f s x) :
    AnalyticWithinAt 𝕜 f s x := by
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
    h : ContDiffWithinAt 𝕜 Top.top f s x
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  obtain ⟨u, hu, hf⟩ := h.analyticOn
  /-
    case intro.intro
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
    h : ContDiffWithinAt 𝕜 Top.top f s x
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    hf : AnalyticOn 𝕜 f u
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  have xu : x ∈ u := mem_of_mem_nhdsWithin (by simp) hu
  /-
    case intro.intro
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
    h : ContDiffWithinAt 𝕜 Top.top f s x
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    hf : AnalyticOn 𝕜 f u
    xu : Membership.mem u x
    ⊢ AnalyticWithinAt 𝕜 f s x
  -/
  exact (hf x xu).mono_of_mem_nhdsWithin (nhdsWithin_mono _ (subset_insert _ _) hu)
  /-
    🎉 no goals
  -/


theorem contDiffWithinAt_omega_iff_analyticWithinAt [CompleteSpace F] :
    ContDiffWithinAt 𝕜 ω f s x ↔ AnalyticWithinAt 𝕜 f s x := by
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type uF
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    inst✝ : CompleteSpace F
    ⊢ Iff (ContDiffWithinAt 𝕜 Top.top f s x) (AnalyticWithinAt 𝕜 f s x)
  -/
  refine ⟨fun h ↦ h.analyticWithinAt, fun h ↦ ?_⟩
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type uF
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    inst✝ : CompleteSpace F
    h : AnalyticWithinAt 𝕜 f s x
    ⊢ ContDiffWithinAt 𝕜 Top.top f s x
  -/
  obtain ⟨u, hu, p, hp, h'p⟩ := h.exists_hasFTaylorSeriesUpToOn ω
  /-
    case intro.intro.intro.intro
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type uF
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    s : Set E
    f : E → F
    x : E
    inst✝ : CompleteSpace F
    h : AnalyticWithinAt 𝕜 f s x
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    p : E → FormalMultilinearSeries 𝕜 E F
    hp : HasFTaylorSeriesUpToOn Top.top f p u
    h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
    ⊢ ContDiffWithinAt 𝕜 Top.top f s x
  -/
  exact ⟨u, hu, p, hp.of_le le_top, fun i ↦ h'p i⟩
  /-
    🎉 no goals
  -/


theorem contDiffWithinAt_nat {n : ℕ} :
    ContDiffWithinAt 𝕜 n f s x ↔ ∃ u ∈ 𝓝[insert x s] x,
      ∃ p : E → FormalMultilinearSeries 𝕜 E F, HasFTaylorSeriesUpToOn n f p u :=
  ⟨fun H => H n le_rfl, fun ⟨u, hu, p, hp⟩ _m hm => ⟨u, hu, p, hp.of_le (mod_cast hm)⟩⟩


/-- When `n` is either a natural number or `ω`, one can characterize the property of being `C^n`
as the existence of a neighborhood on which there is a Taylor series up to order `n`,
requiring in addition that its terms are analytic in the `ω` case. -/
lemma contDiffWithinAt_iff_of_ne_infty (hn : n ≠ ∞) :
    ContDiffWithinAt 𝕜 n f s x ↔ ∃ u ∈ 𝓝[insert x s] x,
      ∃ p : E → FormalMultilinearSeries 𝕜 E F, HasFTaylorSeriesUpToOn n f p u ∧
        (n = ω → ∀ i, AnalyticOn 𝕜 (fun x ↦ p x i) u) := by
  match n with
  | ω => simp [ContDiffWithinAt]
  | ∞ => simp at hn
  | (n : ℕ) => simp [contDiffWithinAt_nat]


theorem ContDiffWithinAt.of_le (h : ContDiffWithinAt 𝕜 n f s x) (hmn : m ≤ n) :
    ContDiffWithinAt 𝕜 m f s x := by
  match n with
  | ω => match m with
    | ω => exact h
    | (m : ℕ∞) =>
      intro k _
      obtain ⟨u, hu, p, hp, -⟩ := h
      exact ⟨u, hu, p, hp.of_le le_top⟩
  | (n : ℕ∞) => match m with
    | ω => simp at hmn
    | (m : ℕ∞) => exact fun k hk ↦ h k (le_trans hk (mod_cast hmn))


/-- In a complete space, a function which is analytic within a set at a point is also `C^ω` there.
Note that the same statement for `AnalyticOn` does not require completeness, see
`AnalyticOn.contDiffOn`. -/
theorem AnalyticWithinAt.contDiffWithinAt [CompleteSpace F] (h : AnalyticWithinAt 𝕜 f s x) :
    ContDiffWithinAt 𝕜 n f s x :=
  (contDiffWithinAt_omega_iff_analyticWithinAt.2 h).of_le le_top


theorem contDiffWithinAt_iff_forall_nat_le {n : ℕ∞} :
    ContDiffWithinAt 𝕜 n f s x ↔ ∀ m : ℕ, ↑m ≤ n → ContDiffWithinAt 𝕜 m f s x :=
  ⟨fun H _ hm => H.of_le (mod_cast hm), fun H m hm => H m hm _ le_rfl⟩


theorem contDiffWithinAt_infty :
    ContDiffWithinAt 𝕜 ∞ f s x ↔ ∀ n : ℕ, ContDiffWithinAt 𝕜 n f s x :=
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
                                                   ⊢ Iff (∀ (m : Nat), LE.le (↑m) Top.top → ContDiffWithinAt 𝕜 (↑m) f s x) (∀ (n  …
                                                 -/
  contDiffWithinAt_iff_forall_nat_le.trans <| by simp only [forall_prop_of_true, le_top]
                                                 /-
                                                   🎉 no goals
                                                 -/


@[deprecated (since := "2024-11-25")] alias contDiffWithinAt_top := contDiffWithinAt_infty


theorem ContDiffWithinAt.continuousWithinAt (h : ContDiffWithinAt 𝕜 n f s x) :
    ContinuousWithinAt f s x := by
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
    h : ContDiffWithinAt 𝕜 n f s x
    ⊢ ContinuousWithinAt f s x
  -/
  have := h.of_le (zero_le _)
  simp only [ContDiffWithinAt, nonpos_iff_eq_zero, Nat.cast_eq_zero,
    mem_pure, forall_eq, CharP.cast_eq_zero] at this
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
    h : ContDiffWithinAt 𝕜 n f s x
    this : Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s))  …
    ⊢ ContinuousWithinAt f s x
  -/
  rcases this with ⟨u, hu, p, H⟩
  /-
    case intro.intro.intro
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
    h : ContDiffWithinAt 𝕜 n f s x
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    p : E → FormalMultilinearSeries 𝕜 E F
    H : HasFTaylorSeriesUpToOn 0 f p u
    ⊢ ContinuousWithinAt f s x
  -/
  rw [mem_nhdsWithin_insert] at hu
  /-
    case intro.intro.intro
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
    h : ContDiffWithinAt 𝕜 n f s x
    u : Set E
    hu : And (Membership.mem u x) (Membership.mem (nhdsWithin x s) u)
    p : E → FormalMultilinearSeries 𝕜 E F
    H : HasFTaylorSeriesUpToOn 0 f p u
    ⊢ ContinuousWithinAt f s x
  -/
  exact (H.continuousOn.continuousWithinAt hu.1).mono_of_mem_nhdsWithin hu.2
  /-
    🎉 no goals
  -/


theorem ContDiffWithinAt.congr_of_eventuallyEq (h : ContDiffWithinAt 𝕜 n f s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) : ContDiffWithinAt 𝕜 n f₁ s x := by
  match n with
  | ω =>
    obtain ⟨u, hu, p, H, H'⟩ := h
    exact ⟨{x ∈ u | f₁ x = f x}, Filter.inter_mem hu (mem_nhdsWithin_insert.2 ⟨hx, h₁⟩), p,
      (H.mono (sep_subset _ _)).congr fun _ ↦ And.right,
      fun i ↦ (H' i).mono (sep_subset _ _)⟩
  | (n : ℕ∞) =>
    intro m hm
    let ⟨u, hu, p, H⟩ := h m hm
    exact ⟨{ x ∈ u | f₁ x = f x }, Filter.inter_mem hu (mem_nhdsWithin_insert.2 ⟨hx, h₁⟩), p,
      (H.mono (sep_subset _ _)).congr fun _ ↦ And.right⟩


theorem Filter.EventuallyEq.congr_contDiffWithinAt (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : f₁ x = f x) :
    ContDiffWithinAt 𝕜 n f₁ s x ↔ ContDiffWithinAt 𝕜 n f s x :=
  ⟨fun H ↦ H.congr_of_eventuallyEq h₁.symm hx.symm, fun H ↦ H.congr_of_eventuallyEq h₁ hx⟩


@[deprecated (since := "2024-10-18")]
alias Filter.EventuallyEq.contDiffWithinAt_iff := Filter.EventuallyEq.congr_contDiffWithinAt


theorem ContDiffWithinAt.congr_of_eventuallyEq_insert (h : ContDiffWithinAt 𝕜 n f s x)
    (h₁ : f₁ =ᶠ[𝓝[insert x s] x] f) : ContDiffWithinAt 𝕜 n f₁ s x :=
  h.congr_of_eventuallyEq (nhdsWithin_mono x (subset_insert x s) h₁)
    (mem_of_mem_nhdsWithin (mem_insert x s) h₁ : _)


theorem Filter.EventuallyEq.congr_contDiffWithinAt_of_insert (h₁ : f₁ =ᶠ[𝓝[insert x s] x] f) :
    ContDiffWithinAt 𝕜 n f₁ s x ↔ ContDiffWithinAt 𝕜 n f s x :=
  ⟨fun H ↦ H.congr_of_eventuallyEq_insert h₁.symm, fun H ↦ H.congr_of_eventuallyEq_insert h₁⟩


theorem ContDiffWithinAt.congr_of_eventuallyEq_of_mem (h : ContDiffWithinAt 𝕜 n f s x)
    (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s) : ContDiffWithinAt 𝕜 n f₁ s x :=
  h.congr_of_eventuallyEq h₁ <| h₁.self_of_nhdsWithin hx


@[deprecated (since := "2024-10-18")]
alias ContDiffWithinAt.congr_of_eventually_eq' := ContDiffWithinAt.congr_of_eventuallyEq_of_mem


theorem Filter.EventuallyEq.congr_contDiffWithinAt_of_mem (h₁ : f₁ =ᶠ[𝓝[s] x] f) (hx : x ∈ s):
    ContDiffWithinAt 𝕜 n f₁ s x ↔ ContDiffWithinAt 𝕜 n f s x :=
  ⟨fun H ↦ H.congr_of_eventuallyEq_of_mem h₁.symm hx, fun H ↦ H.congr_of_eventuallyEq_of_mem h₁ hx⟩


theorem ContDiffWithinAt.congr (h : ContDiffWithinAt 𝕜 n f s x) (h₁ : ∀ y ∈ s, f₁ y = f y)
    (hx : f₁ x = f x) : ContDiffWithinAt 𝕜 n f₁ s x :=
  h.congr_of_eventuallyEq (Filter.eventuallyEq_of_mem self_mem_nhdsWithin h₁) hx


theorem contDiffWithinAt_congr (h₁ : ∀ y ∈ s, f₁ y = f y) (hx : f₁ x = f x) :
    ContDiffWithinAt 𝕜 n f₁ s x ↔ ContDiffWithinAt 𝕜 n f s x :=
  ⟨fun h' ↦ h'.congr (fun x hx ↦ (h₁ x hx).symm) hx.symm, fun h' ↦  h'.congr h₁ hx⟩


theorem ContDiffWithinAt.congr_of_mem (h : ContDiffWithinAt 𝕜 n f s x) (h₁ : ∀ y ∈ s, f₁ y = f y)
    (hx : x ∈ s) : ContDiffWithinAt 𝕜 n f₁ s x :=
  h.congr h₁ (h₁ _ hx)


@[deprecated (since := "2024-10-18")]
alias ContDiffWithinAt.congr' := ContDiffWithinAt.congr_of_mem


theorem contDiffWithinAt_congr_of_mem (h₁ : ∀ y ∈ s, f₁ y = f y) (hx : x ∈ s) :
    ContDiffWithinAt 𝕜 n f₁ s x ↔ ContDiffWithinAt 𝕜 n f s x :=
  contDiffWithinAt_congr h₁ (h₁ x hx)


theorem ContDiffWithinAt.congr_of_insert (h : ContDiffWithinAt 𝕜 n f s x)
    (h₁ : ∀ y ∈ insert x s, f₁ y = f y) : ContDiffWithinAt 𝕜 n f₁ s x :=
  h.congr (fun y hy ↦ h₁ y (mem_insert_of_mem _ hy)) (h₁ x (mem_insert _ _))


theorem contDiffWithinAt_congr_of_insert (h₁ : ∀ y ∈ insert x s, f₁ y = f y) :
    ContDiffWithinAt 𝕜 n f₁ s x ↔ ContDiffWithinAt 𝕜 n f s x :=
  contDiffWithinAt_congr (fun y hy ↦ h₁ y (mem_insert_of_mem _ hy)) (h₁ x (mem_insert _ _))


theorem ContDiffWithinAt.mono_of_mem_nhdsWithin (h : ContDiffWithinAt 𝕜 n f s x) {t : Set E}
    (hst : s ∈ 𝓝[t] x) : ContDiffWithinAt 𝕜 n f t x := by
  match n with
  | ω =>
    obtain ⟨u, hu, p, H, H'⟩ := h
    exact ⟨u, nhdsWithin_le_of_mem (insert_mem_nhdsWithin_insert hst) hu, p, H, H'⟩
  | (n : ℕ∞) =>
    intro m hm
    rcases h m hm with ⟨u, hu, p, H⟩
    exact ⟨u, nhdsWithin_le_of_mem (insert_mem_nhdsWithin_insert hst) hu, p, H⟩


@[deprecated (since := "2024-10-30")]
alias ContDiffWithinAt.mono_of_mem := ContDiffWithinAt.mono_of_mem_nhdsWithin


theorem ContDiffWithinAt.mono (h : ContDiffWithinAt 𝕜 n f s x) {t : Set E} (hst : t ⊆ s) :
    ContDiffWithinAt 𝕜 n f t x :=
  h.mono_of_mem_nhdsWithin <| Filter.mem_of_superset self_mem_nhdsWithin hst


theorem ContDiffWithinAt.congr_mono
    (h : ContDiffWithinAt 𝕜 n f s x) (h' : EqOn f₁ f s₁) (h₁ : s₁ ⊆ s) (hx : f₁ x = f x) :
    ContDiffWithinAt 𝕜 n f₁ s₁ x :=
  (h.mono h₁).congr h' hx


theorem ContDiffWithinAt.congr_set (h : ContDiffWithinAt 𝕜 n f s x) {t : Set E}
    (hst : s =ᶠ[𝓝 x] t) : ContDiffWithinAt 𝕜 n f t x := by
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
    h : ContDiffWithinAt 𝕜 n f s x
    t : Set E
    hst : (nhds x).EventuallyEq s t
    ⊢ ContDiffWithinAt 𝕜 n f t x
  -/
  rw [← nhdsWithin_eq_iff_eventuallyEq] at hst
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
    h : ContDiffWithinAt 𝕜 n f s x
    t : Set E
    hst : Eq (nhdsWithin x s) (nhdsWithin x t)
    ⊢ ContDiffWithinAt 𝕜 n f t x
  -/
  apply h.mono_of_mem_nhdsWithin <| hst ▸ self_mem_nhdsWithin
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-23")]
alias ContDiffWithinAt.congr_nhds := ContDiffWithinAt.congr_set


theorem contDiffWithinAt_congr_set {t : Set E} (hst : s =ᶠ[𝓝 x] t) :
    ContDiffWithinAt 𝕜 n f s x ↔ ContDiffWithinAt 𝕜 n f t x :=
  ⟨fun h => h.congr_set hst, fun h => h.congr_set hst.symm⟩


@[deprecated (since := "2024-10-23")]
alias contDiffWithinAt_congr_nhds := contDiffWithinAt_congr_set


theorem contDiffWithinAt_inter' (h : t ∈ 𝓝[s] x) :
    ContDiffWithinAt 𝕜 n f (s ∩ t) x ↔ ContDiffWithinAt 𝕜 n f s x :=
  contDiffWithinAt_congr_set (mem_nhdsWithin_iff_eventuallyEq.1 h).symm


theorem contDiffWithinAt_inter (h : t ∈ 𝓝 x) :
    ContDiffWithinAt 𝕜 n f (s ∩ t) x ↔ ContDiffWithinAt 𝕜 n f s x :=
  contDiffWithinAt_inter' (mem_nhdsWithin_of_mem_nhds h)


theorem contDiffWithinAt_insert_self :
    ContDiffWithinAt 𝕜 n f (insert x s) x ↔ ContDiffWithinAt 𝕜 n f s x := by
  match n with
  | ω => simp [ContDiffWithinAt]
  | (n : ℕ∞) => simp_rw [ContDiffWithinAt, insert_idem]


theorem contDiffWithinAt_insert {y : E} :
    ContDiffWithinAt 𝕜 n f (insert y s) x ↔ ContDiffWithinAt 𝕜 n f s x := by
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
    y : E
    ⊢ Iff (ContDiffWithinAt 𝕜 n f (Insert.insert y s) x) (ContDiffWithinAt 𝕜 n f s …
  -/
  rcases eq_or_ne x y with (rfl | hx)
    /-
      case inl
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
      ⊢ Iff (ContDiffWithinAt 𝕜 n f (Insert.insert x s) x) (ContDiffWithinAt 𝕜 n f s …
    -/
  · exact contDiffWithinAt_insert_self
    /-
      🎉 no goals
    -/
  /-
    case inr
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
    y : E
    hx : Ne x y
    ⊢ Iff (ContDiffWithinAt 𝕜 n f (Insert.insert y s) x) (ContDiffWithinAt 𝕜 n f s …
  -/
  refine ⟨fun h ↦ h.mono (subset_insert _ _), fun h ↦ ?_⟩
  /-
    case inr
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
    y : E
    hx : Ne x y
    h : ContDiffWithinAt 𝕜 n f s x
    ⊢ ContDiffWithinAt 𝕜 n f (Insert.insert y s) x
  -/
  apply h.mono_of_mem_nhdsWithin
  /-
    case inr
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
    y : E
    hx : Ne x y
    h : ContDiffWithinAt 𝕜 n f s x
    ⊢ Membership.mem (nhdsWithin x (Insert.insert y s)) s
  -/
  simp [nhdsWithin_insert_of_ne hx, self_mem_nhdsWithin]
  /-
    🎉 no goals
  -/


alias ⟨ContDiffWithinAt.of_insert, ContDiffWithinAt.insert'⟩ := contDiffWithinAt_insert


protected theorem ContDiffWithinAt.insert (h : ContDiffWithinAt 𝕜 n f s x) :
    ContDiffWithinAt 𝕜 n f (insert x s) x :=
  h.insert'


theorem contDiffWithinAt_diff_singleton {y : E} :
    ContDiffWithinAt 𝕜 n f (s \ {y}) x ↔ ContDiffWithinAt 𝕜 n f s x := by
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
    y : E
    ⊢ Iff (ContDiffWithinAt 𝕜 n f (SDiff.sdiff s (Singleton.singleton y)) x) (Cont …
  -/
  rw [← contDiffWithinAt_insert, insert_diff_singleton, contDiffWithinAt_insert]
  /-
    🎉 no goals
  -/


/-- If a function is `C^n` within a set at a point, with `n ≥ 1`, then it is differentiable
within this set at this point. -/
theorem ContDiffWithinAt.differentiableWithinAt' (h : ContDiffWithinAt 𝕜 n f s x) (hn : 1 ≤ n) :
    DifferentiableWithinAt 𝕜 f (insert x s) x := by
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
    h : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le 1 n
    ⊢ DifferentiableWithinAt 𝕜 f (Insert.insert x s) x
  -/
  rcases contDiffWithinAt_nat.1 (h.of_le hn) with ⟨u, hu, p, H⟩
  /-
    case intro.intro.intro
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
    h : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le 1 n
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    p : E → FormalMultilinearSeries 𝕜 E F
    H : HasFTaylorSeriesUpToOn (↑One.one) f p u
    ⊢ DifferentiableWithinAt 𝕜 f (Insert.insert x s) x
  -/
  rcases mem_nhdsWithin.1 hu with ⟨t, t_open, xt, tu⟩
  /-
    case intro.intro.intro.intro.intro.intro
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
    h : ContDiffWithinAt 𝕜 n f s x
    hn : LE.le 1 n
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    p : E → FormalMultilinearSeries 𝕜 E F
    H : HasFTaylorSeriesUpToOn (↑One.one) f p u
    t : Set E
    t_open : IsOpen t
    xt : Membership.mem t x
    tu : HasSubset.Subset (Inter.inter t (Insert.insert x s)) u
    ⊢ DifferentiableWithinAt 𝕜 f (Insert.insert x s) x
  -/
  rw [inter_comm] at tu
  exact (differentiableWithinAt_inter (IsOpen.mem_nhds t_open xt)).1 <|
    ((H.mono tu).differentiableOn le_rfl) x ⟨mem_insert x s, xt⟩


@[deprecated (since := "2024-10-10")]
alias ContDiffWithinAt.differentiable_within_at' := ContDiffWithinAt.differentiableWithinAt'


theorem ContDiffWithinAt.differentiableWithinAt (h : ContDiffWithinAt 𝕜 n f s x) (hn : 1 ≤ n) :
    DifferentiableWithinAt 𝕜 f s x :=
  (h.differentiableWithinAt' hn).mono (subset_insert x s)


/-- A function is `C^(n + 1)` on a domain iff locally, it has a derivative which is `C^n`
(and moreover the function is analytic when `n = ω`). -/
theorem contDiffWithinAt_succ_iff_hasFDerivWithinAt (hn : n ≠ ∞) :
    ContDiffWithinAt 𝕜 (n + 1) f s x ↔ ∃ u ∈ 𝓝[insert x s] x, (n = ω → AnalyticOn 𝕜 f u) ∧
      ∃ f' : E → E →L[𝕜] F,
      (∀ x ∈ u, HasFDerivWithinAt f (f' x) u x) ∧ ContDiffWithinAt 𝕜 n f' u x := by
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
    hn : Ne n ↑Top.top
    ⊢ Iff (ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x) (Exists fun u => And (Members …
  -/
  have h'n : n + 1 ≠ ∞ := by simpa using hn
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
    hn : Ne n ↑Top.top
    h'n : Ne (HAdd.hAdd n 1) ↑Top.top
    ⊢ Iff (ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x) (Exists fun u => And (Members …
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
      x : E
      n : WithTop ENat
      hn : Ne n ↑Top.top
      h'n : Ne (HAdd.hAdd n 1) ↑Top.top
      ⊢ ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x → Exists fun u => And (Membership.m …
    -/
  · intro h
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
      x : E
      n : WithTop ENat
      hn : Ne n ↑Top.top
      h'n : Ne (HAdd.hAdd n 1) ↑Top.top
      h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    rcases (contDiffWithinAt_iff_of_ne_infty h'n).1 h with ⟨u, hu, p, Hp, H'p⟩
    refine ⟨u, hu, ?_, fun y => (continuousMultilinearCurryFin1 𝕜 E F) (p y 1),
        fun y hy => Hp.hasFDerivWithinAt le_add_self hy, ?_⟩
      /-
        case mp.intro.intro.intro.intro.refine_1
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        Hp : HasFTaylorSeriesUpToOn (HAdd.hAdd n 1) f p u
        H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
        ⊢ Eq n Top.top → AnalyticOn 𝕜 f u
      -/
    · rintro rfl
      /-
        case mp.intro.intro.intro.intro.refine_1
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
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        hn : Ne Top.top ↑Top.top
        h'n : Ne (HAdd.hAdd Top.top 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd Top.top 1) f s x
        Hp : HasFTaylorSeriesUpToOn (HAdd.hAdd Top.top 1) f p u
        H'p : Eq (HAdd.hAdd Top.top 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p …
        ⊢ AnalyticOn 𝕜 f u
      -/
      exact Hp.analyticOn (H'p rfl 0)
      /-
        🎉 no goals
      -/
    /-
      case mp.intro.intro.intro.intro.refine_2
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
      hn : Ne n ↑Top.top
      h'n : Ne (HAdd.hAdd n 1) ↑Top.top
      h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (HAdd.hAdd n 1) f p u
      H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
      ⊢ ContDiffWithinAt 𝕜 n (fun y => (continuousMultilinearCurryFin1 𝕜 E F) (p y 1 …
    -/
    apply (contDiffWithinAt_iff_of_ne_infty hn).2
    /-
      case mp.intro.intro.intro.intro.refine_2
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
      hn : Ne n ↑Top.top
      h'n : Ne (HAdd.hAdd n 1) ↑Top.top
      h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (HAdd.hAdd n 1) f p u
      H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
      ⊢ Exists fun u_1 => And (Membership.mem (nhdsWithin x (Insert.insert x u)) u_1 …
    -/
    refine ⟨u, ?_, fun y : E => (p y).shift, ?_⟩
    · -- Porting note: without the explicit argument Lean is not sure of the type.
      /-
        case mp.intro.intro.intro.intro.refine_2.refine_1
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        Hp : HasFTaylorSeriesUpToOn (HAdd.hAdd n 1) f p u
        H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
        ⊢ Membership.mem (nhdsWithin x (Insert.insert x u)) u
      -/
      convert @self_mem_nhdsWithin _ _ x u
      /-
        case h.e'_4.h.e'_4
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        Hp : HasFTaylorSeriesUpToOn (HAdd.hAdd n 1) f p u
        H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
        ⊢ Eq (Insert.insert x u) u
      -/
      have : x ∈ insert x s := by simp
      /-
        case h.e'_4.h.e'_4
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        Hp : HasFTaylorSeriesUpToOn (HAdd.hAdd n 1) f p u
        H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
        this : Membership.mem (Insert.insert x s) x
        ⊢ Eq (Insert.insert x u) u
      -/
      exact insert_eq_of_mem (mem_of_mem_nhdsWithin this hu)
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.intro.refine_2.refine_2
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        Hp : HasFTaylorSeriesUpToOn (HAdd.hAdd n 1) f p u
        H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
        ⊢ And (HasFTaylorSeriesUpToOn n (fun y => (continuousMultilinearCurryFin1 𝕜 E  …
      -/
    · rw [hasFTaylorSeriesUpToOn_succ_iff_right] at Hp
      /-
        case mp.intro.intro.intro.intro.refine_2.refine_2
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        Hp : And (∀ (x : E), Membership.mem u x → Eq (p x 0).curry0 (f x)) (And (∀ (x  …
        H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
        ⊢ And (HasFTaylorSeriesUpToOn n (fun y => (continuousMultilinearCurryFin1 𝕜 E  …
      -/
      refine ⟨Hp.2.2, ?_⟩
      /-
        case mp.intro.intro.intro.intro.refine_2.refine_2
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        Hp : And (∀ (x : E), Membership.mem u x → Eq (p x 0).curry0 (f x)) (And (∀ (x  …
        H'p : Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) u
        ⊢ Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => (fun y => (p y).shift) x  …
      -/
      rintro rfl i
      change AnalyticOn 𝕜
        (fun x ↦ (continuousMultilinearCurryRightEquiv' 𝕜 i E F) (p x (i + 1))) u
      apply (LinearIsometryEquiv.analyticOnNhd _ _).comp_analyticOn
        ?_ (Set.mapsTo_univ _ _)
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
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        hn : Ne Top.top ↑Top.top
        h'n : Ne (HAdd.hAdd Top.top 1) ↑Top.top
        h : ContDiffWithinAt 𝕜 (HAdd.hAdd Top.top 1) f s x
        Hp : And (∀ (x : E), Membership.mem u x → Eq (p x 0).curry0 (f x)) (And (∀ (x  …
        H'p : Eq (HAdd.hAdd Top.top 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p …
        i : Nat
        ⊢ AnalyticOn 𝕜 (fun x => p x (HAdd.hAdd i 1)) u
      -/
      exact H'p rfl _
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
      x : E
      n : WithTop ENat
      hn : Ne n ↑Top.top
      h'n : Ne (HAdd.hAdd n 1) ↑Top.top
      ⊢ (Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) ( …
    -/
  · rintro ⟨u, hu, hf, f', f'_eq_deriv, Hf'⟩
    /-
      case mpr.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h'n : Ne (HAdd.hAdd n 1) ↑Top.top
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      hf : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      Hf' : ContDiffWithinAt 𝕜 n f' u x
      ⊢ ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
    -/
    rw [contDiffWithinAt_iff_of_ne_infty h'n]
    /-
      case mpr.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h'n : Ne (HAdd.hAdd n 1) ↑Top.top
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      hf : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      Hf' : ContDiffWithinAt 𝕜 n f' u x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
    -/
    rcases (contDiffWithinAt_iff_of_ne_infty hn).1 Hf' with ⟨v, hv, p', Hp', p'_an⟩
    /-
      case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h'n : Ne (HAdd.hAdd n 1) ↑Top.top
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      hf : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      Hf' : ContDiffWithinAt 𝕜 n f' u x
      v : Set E
      hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
      p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
      Hp' : HasFTaylorSeriesUpToOn n f' p' v
      p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
    -/
    refine ⟨v ∩ u, ?_, fun x => (p' x).unshift (f x), ?_, ?_⟩
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        hf : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        Hf' : ContDiffWithinAt 𝕜 n f' u x
        v : Set E
        hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
        p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
        Hp' : HasFTaylorSeriesUpToOn n f' p' v
        p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
        ⊢ Membership.mem (nhdsWithin x (Insert.insert x s)) (Inter.inter v u)
      -/
    · apply Filter.inter_mem _ hu
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        hf : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        Hf' : ContDiffWithinAt 𝕜 n f' u x
        v : Set E
        hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
        p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
        Hp' : HasFTaylorSeriesUpToOn n f' p' v
        p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
        ⊢ Membership.mem (nhdsWithin x (Insert.insert x s)) v
      -/
      apply nhdsWithin_le_of_mem hu
      /-
        case a
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        hf : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        Hf' : ContDiffWithinAt 𝕜 n f' u x
        v : Set E
        hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
        p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
        Hp' : HasFTaylorSeriesUpToOn n f' p' v
        p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
        ⊢ Membership.mem (nhdsWithin x u) v
      -/
      exact nhdsWithin_mono _ (subset_insert x u) hv
      /-
        🎉 no goals
      -/
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        hf : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        Hf' : ContDiffWithinAt 𝕜 n f' u x
        v : Set E
        hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
        p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
        Hp' : HasFTaylorSeriesUpToOn n f' p' v
        p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
        ⊢ HasFTaylorSeriesUpToOn (HAdd.hAdd n 1) f (fun x => (p' x).unshift (f x)) (In …
      -/
    · rw [hasFTaylorSeriesUpToOn_succ_iff_right]
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        hf : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        Hf' : ContDiffWithinAt 𝕜 n f' u x
        v : Set E
        hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
        p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
        Hp' : HasFTaylorSeriesUpToOn n f' p' v
        p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
        ⊢ And (∀ (x : E), Membership.mem (Inter.inter v u) x → Eq ((p' x).unshift (f x …
      -/
      refine ⟨fun y _ => rfl, fun y hy => ?_, ?_⟩
      · change
          HasFDerivWithinAt (fun z => (continuousMultilinearCurryFin0 𝕜 E F).symm (f z))
            (FormalMultilinearSeries.unshift (p' y) (f y) 1).curryLeft (v ∩ u) y
        -- Porting note: needed `erw` here.
        -- https://github.com/leanprover-community/mathlib4/issues/5164
        /-
          case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_1
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
          hn : Ne n ↑Top.top
          h'n : Ne (HAdd.hAdd n 1) ↑Top.top
          u : Set E
          hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
          hf : Eq n Top.top → AnalyticOn 𝕜 f u
          f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
          f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
          Hf' : ContDiffWithinAt 𝕜 n f' u x
          v : Set E
          hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
          p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
          Hp' : HasFTaylorSeriesUpToOn n f' p' v
          p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
          y : E
          hy : Membership.mem (Inter.inter v u) y
          ⊢ HasFDerivWithinAt (fun z => (continuousMultilinearCurryFin0 𝕜 E F).symm (f z …
        -/
        erw [LinearIsometryEquiv.comp_hasFDerivWithinAt_iff']
        /-
          case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_1
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
          hn : Ne n ↑Top.top
          h'n : Ne (HAdd.hAdd n 1) ↑Top.top
          u : Set E
          hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
          hf : Eq n Top.top → AnalyticOn 𝕜 f u
          f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
          f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
          Hf' : ContDiffWithinAt 𝕜 n f' u x
          v : Set E
          hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
          p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
          Hp' : HasFTaylorSeriesUpToOn n f' p' v
          p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
          y : E
          hy : Membership.mem (Inter.inter v u) y
          ⊢ HasFDerivWithinAt f ((↑{ toLinearEquiv := (continuousMultilinearCurryFin0 𝕜  …
        -/
        convert (f'_eq_deriv y hy.2).mono inter_subset_right
        /-
          case h.e'_12
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
          hn : Ne n ↑Top.top
          h'n : Ne (HAdd.hAdd n 1) ↑Top.top
          u : Set E
          hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
          hf : Eq n Top.top → AnalyticOn 𝕜 f u
          f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
          f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
          Hf' : ContDiffWithinAt 𝕜 n f' u x
          v : Set E
          hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
          p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
          Hp' : HasFTaylorSeriesUpToOn n f' p' v
          p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
          y : E
          hy : Membership.mem (Inter.inter v u) y
          ⊢ Eq ((↑{ toLinearEquiv := (continuousMultilinearCurryFin0 𝕜 E F).symm.symm.to …
        -/
        rw [← Hp'.zero_eq y hy.1]
        /-
          case h.e'_12
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
          hn : Ne n ↑Top.top
          h'n : Ne (HAdd.hAdd n 1) ↑Top.top
          u : Set E
          hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
          hf : Eq n Top.top → AnalyticOn 𝕜 f u
          f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
          f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
          Hf' : ContDiffWithinAt 𝕜 n f' u x
          v : Set E
          hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
          p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
          Hp' : HasFTaylorSeriesUpToOn n f' p' v
          p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
          y : E
          hy : Membership.mem (Inter.inter v u) y
          ⊢ Eq ((↑{ toLinearEquiv := (continuousMultilinearCurryFin0 𝕜 E F).symm.symm.to …
        -/
        ext z
        change ((p' y 0) (init (@cons 0 (fun _ => E) z 0))) (@cons 0 (fun _ => E) z 0 (last 0)) =
          ((p' y 0) 0) z
        /-
          case h.e'_12.h
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
          hn : Ne n ↑Top.top
          h'n : Ne (HAdd.hAdd n 1) ↑Top.top
          u : Set E
          hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
          hf : Eq n Top.top → AnalyticOn 𝕜 f u
          f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
          f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
          Hf' : ContDiffWithinAt 𝕜 n f' u x
          v : Set E
          hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
          p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
          Hp' : HasFTaylorSeriesUpToOn n f' p' v
          p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
          y : E
          hy : Membership.mem (Inter.inter v u) y
          z : E
          ⊢ Eq (((p' y 0) (Fin.init (Fin.cons z 0))) (Fin.cons z 0 (Fin.last 0))) (((p'  …
        -/
        congr
        /-
          case h.e'_12.h.e_a.h.e_6.h
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
          hn : Ne n ↑Top.top
          h'n : Ne (HAdd.hAdd n 1) ↑Top.top
          u : Set E
          hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
          hf : Eq n Top.top → AnalyticOn 𝕜 f u
          f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
          f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
          Hf' : ContDiffWithinAt 𝕜 n f' u x
          v : Set E
          hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
          p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
          Hp' : HasFTaylorSeriesUpToOn n f' p' v
          p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
          y : E
          hy : Membership.mem (Inter.inter v u) y
          z : E
          ⊢ Eq (Fin.init (Fin.cons z 0)) 0
        -/
        norm_num [eq_iff_true_of_subsingleton]
        /-
          🎉 no goals
        -/
        /-
          case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_2.refine_2
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
          hn : Ne n ↑Top.top
          h'n : Ne (HAdd.hAdd n 1) ↑Top.top
          u : Set E
          hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
          hf : Eq n Top.top → AnalyticOn 𝕜 f u
          f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
          f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
          Hf' : ContDiffWithinAt 𝕜 n f' u x
          v : Set E
          hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
          p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
          Hp' : HasFTaylorSeriesUpToOn n f' p' v
          p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
          ⊢ HasFTaylorSeriesUpToOn n (fun x => (continuousMultilinearCurryFin1 𝕜 E F) (( …
        -/
      · convert (Hp'.mono inter_subset_left).congr fun x hx => Hp'.zero_eq x hx.1 using 1
          /-
            case h.e'_10
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
            hn : Ne n ↑Top.top
            h'n : Ne (HAdd.hAdd n 1) ↑Top.top
            u : Set E
            hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
            hf : Eq n Top.top → AnalyticOn 𝕜 f u
            f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
            f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
            Hf' : ContDiffWithinAt 𝕜 n f' u x
            v : Set E
            hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
            p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
            Hp' : HasFTaylorSeriesUpToOn n f' p' v
            p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
            ⊢ Eq (fun x => (continuousMultilinearCurryFin1 𝕜 E F) ((p' x).unshift (f x) 1) …
          -/
        · ext x y
          /-
            case h.e'_10.h.h
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
            x✝ : E
            n : WithTop ENat
            hn : Ne n ↑Top.top
            h'n : Ne (HAdd.hAdd n 1) ↑Top.top
            u : Set E
            hu : Membership.mem (nhdsWithin x✝ (Insert.insert x✝ s)) u
            hf : Eq n Top.top → AnalyticOn 𝕜 f u
            f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
            f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
            Hf' : ContDiffWithinAt 𝕜 n f' u x✝
            v : Set E
            hv : Membership.mem (nhdsWithin x✝ (Insert.insert x✝ u)) v
            p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
            Hp' : HasFTaylorSeriesUpToOn n f' p' v
            p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
            x y : E
            ⊢ Eq (((continuousMultilinearCurryFin1 𝕜 E F) ((p' x).unshift (f x) 1)) y) ((p …
          -/
          change p' x 0 (init (@snoc 0 (fun _ : Fin 1 => E) 0 y)) y = p' x 0 0 y
          /-
            case h.e'_10.h.h
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
            x✝ : E
            n : WithTop ENat
            hn : Ne n ↑Top.top
            h'n : Ne (HAdd.hAdd n 1) ↑Top.top
            u : Set E
            hu : Membership.mem (nhdsWithin x✝ (Insert.insert x✝ s)) u
            hf : Eq n Top.top → AnalyticOn 𝕜 f u
            f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
            f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
            Hf' : ContDiffWithinAt 𝕜 n f' u x✝
            v : Set E
            hv : Membership.mem (nhdsWithin x✝ (Insert.insert x✝ u)) v
            p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
            Hp' : HasFTaylorSeriesUpToOn n f' p' v
            p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
            x y : E
            ⊢ Eq (((p' x 0) (Fin.init (Fin.snoc 0 y))) y) (((p' x 0) 0) y)
          -/
          rw [init_snoc]
          /-
            🎉 no goals
          -/
          /-
            case h.e'_11
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
            hn : Ne n ↑Top.top
            h'n : Ne (HAdd.hAdd n 1) ↑Top.top
            u : Set E
            hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
            hf : Eq n Top.top → AnalyticOn 𝕜 f u
            f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
            f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
            Hf' : ContDiffWithinAt 𝕜 n f' u x
            v : Set E
            hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
            p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
            Hp' : HasFTaylorSeriesUpToOn n f' p' v
            p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
            ⊢ Eq (fun x => ((p' x).unshift (f x)).shift) p'
          -/
        · ext x k v y
          change p' x k (init (@snoc k (fun _ : Fin k.succ => E) v y))
            (@snoc k (fun _ : Fin k.succ => E) v y (last k)) = p' x k v y
          /-
            case h.e'_11.h.h.H.h
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
            x✝ : E
            n : WithTop ENat
            hn : Ne n ↑Top.top
            h'n : Ne (HAdd.hAdd n 1) ↑Top.top
            u : Set E
            hu : Membership.mem (nhdsWithin x✝ (Insert.insert x✝ s)) u
            hf : Eq n Top.top → AnalyticOn 𝕜 f u
            f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
            f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
            Hf' : ContDiffWithinAt 𝕜 n f' u x✝
            v✝ : Set E
            hv : Membership.mem (nhdsWithin x✝ (Insert.insert x✝ u)) v✝
            p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
            Hp' : HasFTaylorSeriesUpToOn n f' p' v✝
            p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v✝
            x : E
            k : Nat
            v : Fin k → E
            y : E
            ⊢ Eq (((p' x k) (Fin.init (Fin.snoc v y))) (Fin.snoc v y (Fin.last k))) (((p'  …
          -/
          rw [snoc_last, init_snoc]
          /-
            🎉 no goals
          -/
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_3
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        hf : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        Hf' : ContDiffWithinAt 𝕜 n f' u x
        v : Set E
        hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
        p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
        Hp' : HasFTaylorSeriesUpToOn n f' p' v
        p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
        ⊢ Eq (HAdd.hAdd n 1) Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => (fun x => ( …
      -/
    · intro h i
      /-
        case mpr.intro.intro.intro.intro.intro.intro.intro.intro.intro.refine_3
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
        hn : Ne n ↑Top.top
        h'n : Ne (HAdd.hAdd n 1) ↑Top.top
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        hf : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        f'_eq_deriv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        Hf' : ContDiffWithinAt 𝕜 n f' u x
        v : Set E
        hv : Membership.mem (nhdsWithin x (Insert.insert x u)) v
        p' : E → FormalMultilinearSeries 𝕜 E (ContinuousLinearMap (RingHom.id 𝕜) E F)
        Hp' : HasFTaylorSeriesUpToOn n f' p' v
        p'_an : Eq n Top.top → ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p' x i) v
        h : Eq (HAdd.hAdd n 1) Top.top
        i : Nat
        ⊢ AnalyticOn 𝕜 (fun x => (fun x => (p' x).unshift (f x)) x i) (Inter.inter v u)
      -/
      simp only [WithTop.add_eq_top, WithTop.one_ne_top, or_false] at h
      match i with
      | 0 =>
        simp only [FormalMultilinearSeries.unshift]
        apply AnalyticOnNhd.comp_analyticOn _ ((hf h).mono inter_subset_right)
          (Set.mapsTo_univ _ _)
        exact LinearIsometryEquiv.analyticOnNhd _ _
      | i + 1 =>
        simp only [FormalMultilinearSeries.unshift, Nat.succ_eq_add_one]
        apply AnalyticOnNhd.comp_analyticOn _ ((p'_an h i).mono inter_subset_left)
          (Set.mapsTo_univ _ _)
        exact LinearIsometryEquiv.analyticOnNhd _ _


/-- A version of `contDiffWithinAt_succ_iff_hasFDerivWithinAt` where all derivatives
  are taken within the same set. -/
theorem contDiffWithinAt_succ_iff_hasFDerivWithinAt' (hn : n ≠ ∞) :
    ContDiffWithinAt 𝕜 (n + 1) f s x ↔
      ∃ u ∈ 𝓝[insert x s] x, u ⊆ insert x s ∧ (n = ω → AnalyticOn 𝕜 f u) ∧
      ∃ f' : E → E →L[𝕜] F,
        (∀ x ∈ u, HasFDerivWithinAt f (f' x) s x) ∧ ContDiffWithinAt 𝕜 n f' s x := by
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
    hn : Ne n ↑Top.top
    ⊢ Iff (ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x) (Exists fun u => And (Members …
  -/
  refine ⟨fun hf => ?_, ?_⟩
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
      x : E
      n : WithTop ENat
      hn : Ne n ↑Top.top
      hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
  · obtain ⟨u, hu, f_an, f', huf', hf'⟩ := (contDiffWithinAt_succ_iff_hasFDerivWithinAt hn).mp hf
    /-
      case refine_1.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      hf' : ContDiffWithinAt 𝕜 n f' u x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    obtain ⟨w, hw, hxw, hwu⟩ := mem_nhdsWithin.mp hu
    /-
      case refine_1.intro.intro.intro.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      hf' : ContDiffWithinAt 𝕜 n f' u x
      w : Set E
      hw : IsOpen w
      hxw : Membership.mem w x
      hwu : HasSubset.Subset (Inter.inter w (Insert.insert x s)) u
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    rw [inter_comm] at hwu
    refine ⟨insert x s ∩ w, inter_mem_nhdsWithin _ (hw.mem_nhds hxw), inter_subset_left, ?_, f',
      fun y hy => ?_, ?_⟩
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
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
        hn : Ne n ↑Top.top
        hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        f_an : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        hf' : ContDiffWithinAt 𝕜 n f' u x
        w : Set E
        hw : IsOpen w
        hxw : Membership.mem w x
        hwu : HasSubset.Subset (Inter.inter (Insert.insert x s) w) u
        ⊢ Eq n Top.top → AnalyticOn 𝕜 f (Inter.inter (Insert.insert x s) w)
      -/
    · intro h
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.refine_1
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
        hn : Ne n ↑Top.top
        hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        f_an : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        hf' : ContDiffWithinAt 𝕜 n f' u x
        w : Set E
        hw : IsOpen w
        hxw : Membership.mem w x
        hwu : HasSubset.Subset (Inter.inter (Insert.insert x s) w) u
        h : Eq n Top.top
        ⊢ AnalyticOn 𝕜 f (Inter.inter (Insert.insert x s) w)
      -/
      apply (f_an h).mono hwu
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
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
        hn : Ne n ↑Top.top
        hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        f_an : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        hf' : ContDiffWithinAt 𝕜 n f' u x
        w : Set E
        hw : IsOpen w
        hxw : Membership.mem w x
        hwu : HasSubset.Subset (Inter.inter (Insert.insert x s) w) u
        y : E
        hy : Membership.mem (Inter.inter (Insert.insert x s) w) y
        ⊢ HasFDerivWithinAt f (f' y) s y
      -/
    · refine ((huf' y <| hwu hy).mono hwu).mono_of_mem_nhdsWithin ?_
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
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
        hn : Ne n ↑Top.top
        hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        f_an : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        hf' : ContDiffWithinAt 𝕜 n f' u x
        w : Set E
        hw : IsOpen w
        hxw : Membership.mem w x
        hwu : HasSubset.Subset (Inter.inter (Insert.insert x s) w) u
        y : E
        hy : Membership.mem (Inter.inter (Insert.insert x s) w) y
        ⊢ Membership.mem (nhdsWithin y s) (Inter.inter (Insert.insert x s) w)
      -/
      refine mem_of_superset ?_ (inter_subset_inter_left _ (subset_insert _ _))
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.refine_2
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
        hn : Ne n ↑Top.top
        hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        f_an : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        hf' : ContDiffWithinAt 𝕜 n f' u x
        w : Set E
        hw : IsOpen w
        hxw : Membership.mem w x
        hwu : HasSubset.Subset (Inter.inter (Insert.insert x s) w) u
        y : E
        hy : Membership.mem (Inter.inter (Insert.insert x s) w) y
        ⊢ Membership.mem (nhdsWithin y s) (Inter.inter s w)
      -/
      exact inter_mem_nhdsWithin _ (hw.mem_nhds hy.2)
      /-
        🎉 no goals
      -/
      /-
        case refine_1.intro.intro.intro.intro.intro.intro.intro.intro.refine_3
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
        hn : Ne n ↑Top.top
        hf : ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
        u : Set E
        hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        f_an : Eq n Top.top → AnalyticOn 𝕜 f u
        f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
        huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
        hf' : ContDiffWithinAt 𝕜 n f' u x
        w : Set E
        hw : IsOpen w
        hxw : Membership.mem w x
        hwu : HasSubset.Subset (Inter.inter (Insert.insert x s) w) u
        ⊢ ContDiffWithinAt 𝕜 n f' s x
      -/
    · exact hf'.mono_of_mem_nhdsWithin (nhdsWithin_mono _ (subset_insert _ _) hu)
      /-
        🎉 no goals
      -/
  · rw [← contDiffWithinAt_insert, contDiffWithinAt_succ_iff_hasFDerivWithinAt hn,
      insert_eq_of_mem (mem_insert _ _)]
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
      x : E
      n : WithTop ENat
      hn : Ne n ↑Top.top
      ⊢ (Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) ( …
    -/
    rintro ⟨u, hu, hus, f_an, f', huf', hf'⟩
    /-
      case refine_2.intro.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      hus : HasSubset.Subset u (Insert.insert x s)
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      huf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) s x
      hf' : ContDiffWithinAt 𝕜 n f' s x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    exact ⟨u, hu, f_an, f', fun y hy => (huf' y hy).insert'.mono hus, hf'.insert.mono hus⟩
    /-
      🎉 no goals
    -/



variable (𝕜) in
/-- A function is continuously differentiable up to `n` on `s` if, for any point `x` in `s`, it
admits continuous derivatives up to order `n` on a neighborhood of `x` in `s`.

For `n = ∞`, we only require that this holds up to any finite order (where the neighborhood may
depend on the finite order we consider).
-/
def ContDiffOn (n : WithTop ℕ∞) (f : E → F) (s : Set E) : Prop :=
  ∀ x ∈ s, ContDiffWithinAt 𝕜 n f s x


theorem HasFTaylorSeriesUpToOn.contDiffOn {n : ℕ∞} {f' : E → FormalMultilinearSeries 𝕜 E F}
    (hf : HasFTaylorSeriesUpToOn n f f' s) : ContDiffOn 𝕜 n f s := by
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
    n : ENat
    f' : E → FormalMultilinearSeries 𝕜 E F
    hf : HasFTaylorSeriesUpToOn (↑n) f f' s
    ⊢ ContDiffOn 𝕜 (↑n) f s
  -/
  intro x hx m hm
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
    n : ENat
    f' : E → FormalMultilinearSeries 𝕜 E F
    hf : HasFTaylorSeriesUpToOn (↑n) f f' s
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) n
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
  -/
  use s
  /-
    case h
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
    n : ENat
    f' : E → FormalMultilinearSeries 𝕜 E F
    hf : HasFTaylorSeriesUpToOn (↑n) f f' s
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) n
    ⊢ And (Membership.mem (nhdsWithin x (Insert.insert x s)) s) (Exists fun p => H …
  -/
  simp only [Set.insert_eq_of_mem hx, self_mem_nhdsWithin, true_and]
  /-
    case h
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
    n : ENat
    f' : E → FormalMultilinearSeries 𝕜 E F
    hf : HasFTaylorSeriesUpToOn (↑n) f f' s
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) n
    ⊢ Exists fun p => HasFTaylorSeriesUpToOn (↑m) f p s
  -/
  exact ⟨f', hf.of_le (mod_cast hm)⟩
  /-
    🎉 no goals
  -/


theorem ContDiffOn.contDiffWithinAt (h : ContDiffOn 𝕜 n f s) (hx : x ∈ s) :
    ContDiffWithinAt 𝕜 n f s x :=
  h x hx


theorem ContDiffOn.of_le (h : ContDiffOn 𝕜 n f s) (hmn : m ≤ n) : ContDiffOn 𝕜 m f s := fun x hx =>
  (h x hx).of_le hmn


theorem ContDiffWithinAt.contDiffOn' (hm : m ≤ n) (h' : m = ∞ → n = ω)
    (h : ContDiffWithinAt 𝕜 n f s x) :
    ∃ u, IsOpen u ∧ x ∈ u ∧ ContDiffOn 𝕜 m f (insert x s ∩ u) := by
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
    m n : WithTop ENat
    hm : LE.le m n
    h' : Eq m ↑Top.top → Eq n Top.top
    h : ContDiffWithinAt 𝕜 n f s x
    ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContDiffOn 𝕜 m f ( …
  -/
  rcases eq_or_ne n ω with rfl | hn
    /-
      case inl
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
      m : WithTop ENat
      hm : LE.le m Top.top
      h' : Eq m ↑Top.top → Eq Top.top Top.top
      h : ContDiffWithinAt 𝕜 Top.top f s x
      ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContDiffOn 𝕜 m f ( …
    -/
  · obtain ⟨t, ht, p, hp, h'p⟩ := h
    /-
      case inl.intro.intro.intro.intro
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
      m : WithTop ENat
      hm : LE.le m Top.top
      h' : Eq m ↑Top.top → Eq Top.top Top.top
      t : Set E
      ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
      p : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn Top.top f p t
      h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) t
      ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContDiffOn 𝕜 m f ( …
    -/
    rcases mem_nhdsWithin.1 ht with ⟨u, huo, hxu, hut⟩
    /-
      case inl.intro.intro.intro.intro.intro.intro.intro
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
      m : WithTop ENat
      hm : LE.le m Top.top
      h' : Eq m ↑Top.top → Eq Top.top Top.top
      t : Set E
      ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
      p : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn Top.top f p t
      h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) t
      u : Set E
      huo : IsOpen u
      hxu : Membership.mem u x
      hut : HasSubset.Subset (Inter.inter u (Insert.insert x s)) t
      ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContDiffOn 𝕜 m f ( …
    -/
    rw [inter_comm] at hut
    /-
      case inl.intro.intro.intro.intro.intro.intro.intro
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
      m : WithTop ENat
      hm : LE.le m Top.top
      h' : Eq m ↑Top.top → Eq Top.top Top.top
      t : Set E
      ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
      p : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn Top.top f p t
      h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) t
      u : Set E
      huo : IsOpen u
      hxu : Membership.mem u x
      hut : HasSubset.Subset (Inter.inter (Insert.insert x s) u) t
      ⊢ Exists fun u => And (IsOpen u) (And (Membership.mem u x) (ContDiffOn 𝕜 m f ( …
    -/
    refine ⟨u, huo, hxu, ?_⟩
    /-
      case inl.intro.intro.intro.intro.intro.intro.intro
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
      m : WithTop ENat
      hm : LE.le m Top.top
      h' : Eq m ↑Top.top → Eq Top.top Top.top
      t : Set E
      ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
      p : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn Top.top f p t
      h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) t
      u : Set E
      huo : IsOpen u
      hxu : Membership.mem u x
      hut : HasSubset.Subset (Inter.inter (Insert.insert x s) u) t
      ⊢ ContDiffOn 𝕜 m f (Inter.inter (Insert.insert x s) u)
    -/
    suffices ContDiffOn 𝕜 ω f (insert x s ∩ u) from this.of_le le_top
    /-
      case inl.intro.intro.intro.intro.intro.intro.intro
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
      m : WithTop ENat
      hm : LE.le m Top.top
      h' : Eq m ↑Top.top → Eq Top.top Top.top
      t : Set E
      ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
      p : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn Top.top f p t
      h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) t
      u : Set E
      huo : IsOpen u
      hxu : Membership.mem u x
      hut : HasSubset.Subset (Inter.inter (Insert.insert x s) u) t
      ⊢ ContDiffOn 𝕜 Top.top f (Inter.inter (Insert.insert x s) u)
    -/
    intro y hy
    /-
      case inl.intro.intro.intro.intro.intro.intro.intro
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
      m : WithTop ENat
      hm : LE.le m Top.top
      h' : Eq m ↑Top.top → Eq Top.top Top.top
      t : Set E
      ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
      p : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn Top.top f p t
      h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) t
      u : Set E
      huo : IsOpen u
      hxu : Membership.mem u x
      hut : HasSubset.Subset (Inter.inter (Insert.insert x s) u) t
      y : E
      hy : Membership.mem (Inter.inter (Insert.insert x s) u) y
      ⊢ ContDiffWithinAt 𝕜 Top.top f (Inter.inter (Insert.insert x s) u) y
    -/
    refine ⟨insert x s ∩ u, ?_, p, hp.mono hut,  fun i ↦ (h'p i).mono hut⟩
    /-
      case inl.intro.intro.intro.intro.intro.intro.intro
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
      m : WithTop ENat
      hm : LE.le m Top.top
      h' : Eq m ↑Top.top → Eq Top.top Top.top
      t : Set E
      ht : Membership.mem (nhdsWithin x (Insert.insert x s)) t
      p : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn Top.top f p t
      h'p : ∀ (i : Nat), AnalyticOn 𝕜 (fun x => p x i) t
      u : Set E
      huo : IsOpen u
      hxu : Membership.mem u x
      hut : HasSubset.Subset (Inter.inter (Insert.insert x s) u) t
      y : E
      hy : Membership.mem (Inter.inter (Insert.insert x s) u) y
      ⊢ Membership.mem (nhdsWithin y (Insert.insert y (Inter.inter (Insert.insert x  …
    -/
    simp only [insert_eq_of_mem, hy, self_mem_nhdsWithin]
    /-
      🎉 no goals
    -/
  · match m with
    | ω => simp [hn] at hm
    | ∞ => exact (hn (h' rfl)).elim
    | (m : ℕ) =>
      rcases contDiffWithinAt_nat.1 (h.of_le hm) with ⟨t, ht, p, hp⟩
      rcases mem_nhdsWithin.1 ht with ⟨u, huo, hxu, hut⟩
      rw [inter_comm] at hut
      exact ⟨u, huo, hxu, (hp.mono hut).contDiffOn⟩


theorem ContDiffWithinAt.contDiffOn (hm : m ≤ n) (h' : m = ∞ → n = ω)
    (h : ContDiffWithinAt 𝕜 n f s x) :
    ∃ u ∈ 𝓝[insert x s] x, u ⊆ insert x s ∧ ContDiffOn 𝕜 m f u := by
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
    m n : WithTop ENat
    hm : LE.le m n
    h' : Eq m ↑Top.top → Eq n Top.top
    h : ContDiffWithinAt 𝕜 n f s x
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
  -/
  obtain ⟨_u, uo, xu, h⟩ := h.contDiffOn' hm h'
  /-
    case intro.intro.intro
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
    m n : WithTop ENat
    hm : LE.le m n
    h' : Eq m ↑Top.top → Eq n Top.top
    h✝ : ContDiffWithinAt 𝕜 n f s x
    _u : Set E
    uo : IsOpen _u
    xu : Membership.mem _u x
    h : ContDiffOn 𝕜 m f (Inter.inter (Insert.insert x s) _u)
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
  -/
  exact ⟨_, inter_mem_nhdsWithin _ (uo.mem_nhds xu), inter_subset_left, h⟩
  /-
    🎉 no goals
  -/


theorem ContDiffOn.analyticOn (h : ContDiffOn 𝕜 ω f s) : AnalyticOn 𝕜 f s :=
  fun x hx ↦ (h x hx).analyticWithinAt


/-- A function is `C^n` within a set at a point, for `n : ℕ`, if and only if it is `C^n` on
a neighborhood of this point. -/
theorem contDiffWithinAt_iff_contDiffOn_nhds (hn : n ≠ ∞) :
    ContDiffWithinAt 𝕜 n f s x ↔ ∃ u ∈ 𝓝[insert x s] x, ContDiffOn 𝕜 n f u := by
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
    hn : Ne n ↑Top.top
    ⊢ Iff (ContDiffWithinAt 𝕜 n f s x) (Exists fun u => And (Membership.mem (nhdsW …
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
      x : E
      n : WithTop ENat
      hn : Ne n ↑Top.top
      h : ContDiffWithinAt 𝕜 n f s x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (C …
    -/
  · rcases h.contDiffOn le_rfl (by simp [hn]) with ⟨u, hu, h'u⟩
    /-
      case refine_1.intro.intro
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
      hn : Ne n ↑Top.top
      h : ContDiffWithinAt 𝕜 n f s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      h'u : And (HasSubset.Subset u (Insert.insert x s)) (ContDiffOn 𝕜 n f u)
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (C …
    -/
    exact ⟨u, hu, h'u.2⟩
    /-
      🎉 no goals
    -/
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
      x : E
      n : WithTop ENat
      hn : Ne n ↑Top.top
      h : Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u)  …
      ⊢ ContDiffWithinAt 𝕜 n f s x
    -/
  · rcases h with ⟨u, u_mem, hu⟩
    /-
      case refine_2.intro.intro
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
      hn : Ne n ↑Top.top
      u : Set E
      u_mem : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      hu : ContDiffOn 𝕜 n f u
      ⊢ ContDiffWithinAt 𝕜 n f s x
    -/
    have : x ∈ u := mem_of_mem_nhdsWithin (mem_insert x s) u_mem
    /-
      case refine_2.intro.intro
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
      hn : Ne n ↑Top.top
      u : Set E
      u_mem : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      hu : ContDiffOn 𝕜 n f u
      this : Membership.mem u x
      ⊢ ContDiffWithinAt 𝕜 n f s x
    -/
    exact (hu x this).mono_of_mem_nhdsWithin (nhdsWithin_mono _ (subset_insert x s) u_mem)
    /-
      🎉 no goals
    -/


protected theorem ContDiffWithinAt.eventually (h : ContDiffWithinAt 𝕜 n f s x) (hn : n ≠ ∞) :
    ∀ᶠ y in 𝓝[insert x s] x, ContDiffWithinAt 𝕜 n f s y := by
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
    h : ContDiffWithinAt 𝕜 n f s x
    hn : Ne n ↑Top.top
    ⊢ Filter.Eventually (fun y => ContDiffWithinAt 𝕜 n f s y) (nhdsWithin x (Inser …
  -/
  rcases h.contDiffOn le_rfl (by simp [hn]) with ⟨u, hu, _, hd⟩
  have : ∀ᶠ y : E in 𝓝[insert x s] x, u ∈ 𝓝[insert x s] y ∧ y ∈ u :=
    (eventually_eventually_nhdsWithin.2 hu).and hu
  /-
    case intro.intro.intro
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
    h : ContDiffWithinAt 𝕜 n f s x
    hn : Ne n ↑Top.top
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    left✝ : HasSubset.Subset u (Insert.insert x s)
    hd : ContDiffOn 𝕜 n f u
    this : Filter.Eventually (fun y => And (Membership.mem (nhdsWithin y (Insert.i …
    ⊢ Filter.Eventually (fun y => ContDiffWithinAt 𝕜 n f s y) (nhdsWithin x (Inser …
  -/
  refine this.mono fun y hy => (hd y hy.2).mono_of_mem_nhdsWithin ?_
  /-
    case intro.intro.intro
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
    h : ContDiffWithinAt 𝕜 n f s x
    hn : Ne n ↑Top.top
    u : Set E
    hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
    left✝ : HasSubset.Subset u (Insert.insert x s)
    hd : ContDiffOn 𝕜 n f u
    this : Filter.Eventually (fun y => And (Membership.mem (nhdsWithin y (Insert.i …
    y : E
    hy : And (Membership.mem (nhdsWithin y (Insert.insert x s)) u) (Membership.mem …
    ⊢ Membership.mem (nhdsWithin y s) u
  -/
  exact nhdsWithin_mono y (subset_insert _ _) hy.1
  /-
    🎉 no goals
  -/


theorem ContDiffOn.of_succ (h : ContDiffOn 𝕜 (n + 1) f s) : ContDiffOn 𝕜 n f s :=
  h.of_le le_self_add


theorem ContDiffOn.one_of_succ (h : ContDiffOn 𝕜 (n + 1) f s) : ContDiffOn 𝕜 1 f s :=
  h.of_le le_add_self


theorem contDiffOn_iff_forall_nat_le {n : ℕ∞} :
    ContDiffOn 𝕜 n f s ↔ ∀ m : ℕ, ↑m ≤ n → ContDiffOn 𝕜 m f s :=
  ⟨fun H _ hm => H.of_le (mod_cast hm), fun H x hx m hm => H m hm x hx m le_rfl⟩


theorem contDiffOn_infty : ContDiffOn 𝕜 ∞ f s ↔ ∀ n : ℕ, ContDiffOn 𝕜 n f s :=
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
                                             ⊢ Iff (∀ (m : Nat), LE.le (↑m) Top.top → ContDiffOn 𝕜 (↑m) f s) (∀ (n : Nat),  …
                                           -/
  contDiffOn_iff_forall_nat_le.trans <| by simp only [le_top, forall_prop_of_true]
                                           /-
                                             🎉 no goals
                                           -/


@[deprecated (since := "2024-11-27")] alias contDiffOn_top := contDiffOn_infty

@[deprecated (since := "2024-11-27")]
alias contDiffOn_infty_iff_contDiffOn_omega := contDiffOn_infty


theorem contDiffOn_all_iff_nat :
    (∀ (n : ℕ∞), ContDiffOn 𝕜 n f s) ↔ ∀ n : ℕ, ContDiffOn 𝕜 n f s := by
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
    ⊢ Iff (∀ (n : ENat), ContDiffOn 𝕜 (↑n) f s) (∀ (n : Nat), ContDiffOn 𝕜 (↑n) f s)
  -/
  refine ⟨fun H n => H n, ?_⟩
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
    ⊢ (∀ (n : Nat), ContDiffOn 𝕜 (↑n) f s) → ∀ (n : ENat), ContDiffOn 𝕜 (↑n) f s
  -/
  rintro H (_ | n)
  /-
    case none
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
    H : ∀ (n : Nat), ContDiffOn 𝕜 (↑n) f s
    ⊢ ContDiffOn 𝕜 (↑Option.none) f s
  -/
  exacts [contDiffOn_infty.2 H, H n]
  /-
    🎉 no goals
  -/


theorem ContDiffOn.continuousOn (h : ContDiffOn 𝕜 n f s) : ContinuousOn f s := fun x hx =>
  (h x hx).continuousWithinAt


theorem ContDiffOn.congr (h : ContDiffOn 𝕜 n f s) (h₁ : ∀ x ∈ s, f₁ x = f x) :
    ContDiffOn 𝕜 n f₁ s := fun x hx => (h x hx).congr h₁ (h₁ x hx)


theorem contDiffOn_congr (h₁ : ∀ x ∈ s, f₁ x = f x) : ContDiffOn 𝕜 n f₁ s ↔ ContDiffOn 𝕜 n f s :=
  ⟨fun H => H.congr fun x hx => (h₁ x hx).symm, fun H => H.congr h₁⟩


theorem ContDiffOn.mono (h : ContDiffOn 𝕜 n f s) {t : Set E} (hst : t ⊆ s) : ContDiffOn 𝕜 n f t :=
  fun x hx => (h x (hst hx)).mono hst


theorem ContDiffOn.congr_mono (hf : ContDiffOn 𝕜 n f s) (h₁ : ∀ x ∈ s₁, f₁ x = f x) (hs : s₁ ⊆ s) :
    ContDiffOn 𝕜 n f₁ s₁ :=
  (hf.mono hs).congr h₁


/-- If a function is `C^n` on a set with `n ≥ 1`, then it is differentiable there. -/
theorem ContDiffOn.differentiableOn (h : ContDiffOn 𝕜 n f s) (hn : 1 ≤ n) :
    DifferentiableOn 𝕜 f s := fun x hx => (h x hx).differentiableWithinAt hn


/-- If a function is `C^n` around each point in a set, then it is `C^n` on the set. -/
theorem contDiffOn_of_locally_contDiffOn
    (h : ∀ x ∈ s, ∃ u, IsOpen u ∧ x ∈ u ∧ ContDiffOn 𝕜 n f (s ∩ u)) : ContDiffOn 𝕜 n f s := by
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
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    ⊢ ContDiffOn 𝕜 n f s
  -/
  intro x xs
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
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : E
    xs : Membership.mem s x
    ⊢ ContDiffWithinAt 𝕜 n f s x
  -/
  rcases h x xs with ⟨u, u_open, xu, hu⟩
  /-
    case intro.intro.intro
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
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : E
    xs : Membership.mem s x
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 n f (Inter.inter s u)
    ⊢ ContDiffWithinAt 𝕜 n f s x
  -/
  apply (contDiffWithinAt_inter _).1 (hu x ⟨xs, xu⟩)
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
    h : ∀ (x : E), Membership.mem s x → Exists fun u => And (IsOpen u) (And (Membe …
    x : E
    xs : Membership.mem s x
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 n f (Inter.inter s u)
    ⊢ Membership.mem (nhds x) u
  -/
  exact IsOpen.mem_nhds u_open xu
  /-
    🎉 no goals
  -/


/-- A function is `C^(n + 1)` on a domain iff locally, it has a derivative which is `C^n`. -/
theorem contDiffOn_succ_iff_hasFDerivWithinAt (hn : n ≠ ∞) :
    ContDiffOn 𝕜 (n + 1) f s ↔
      ∀ x ∈ s, ∃ u ∈ 𝓝[insert x s] x, (n = ω → AnalyticOn 𝕜 f u) ∧ ∃ f' : E → E →L[𝕜] F,
        (∀ x ∈ u, HasFDerivWithinAt f (f' x) u x) ∧ ContDiffOn 𝕜 n f' u := by
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
    hn : Ne n ↑Top.top
    ⊢ Iff (ContDiffOn 𝕜 (HAdd.hAdd n 1) f s) (∀ (x : E), Membership.mem s x → Exis …
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
      n : WithTop ENat
      hn : Ne n ↑Top.top
      ⊢ ContDiffOn 𝕜 (HAdd.hAdd n 1) f s → ∀ (x : E), Membership.mem s x → Exists fu …
    -/
  · intro h x hx
    rcases (contDiffWithinAt_succ_iff_hasFDerivWithinAt hn).1 (h x hx) with
      ⟨u, hu, f_an, f', hf', Hf'⟩
    /-
      case mp.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h : ContDiffOn 𝕜 (HAdd.hAdd n 1) f s
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      Hf' : ContDiffWithinAt 𝕜 n f' u x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    rcases Hf'.contDiffOn le_rfl (by simp [hn]) with ⟨v, vu, v'u, hv⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h : ContDiffOn 𝕜 (HAdd.hAdd n 1) f s
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      Hf' : ContDiffWithinAt 𝕜 n f' u x
      v : Set E
      vu : Membership.mem (nhdsWithin x (Insert.insert x u)) v
      v'u : HasSubset.Subset v (Insert.insert x u)
      hv : ContDiffOn 𝕜 n f' v
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    rw [insert_eq_of_mem hx] at hu ⊢
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h : ContDiffOn 𝕜 (HAdd.hAdd n 1) f s
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x s) u
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      Hf' : ContDiffWithinAt 𝕜 n f' u x
      v : Set E
      vu : Membership.mem (nhdsWithin x (Insert.insert x u)) v
      v'u : HasSubset.Subset v (Insert.insert x u)
      hv : ContDiffOn 𝕜 n f' v
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (And (Eq n Top.top → …
    -/
    have xu : x ∈ u := mem_of_mem_nhdsWithin hx hu
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h : ContDiffOn 𝕜 (HAdd.hAdd n 1) f s
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x s) u
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hf' : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      Hf' : ContDiffWithinAt 𝕜 n f' u x
      v : Set E
      vu : Membership.mem (nhdsWithin x (Insert.insert x u)) v
      v'u : HasSubset.Subset v (Insert.insert x u)
      hv : ContDiffOn 𝕜 n f' v
      xu : Membership.mem u x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (And (Eq n Top.top → …
    -/
    rw [insert_eq_of_mem xu] at vu v'u
    exact ⟨v, nhdsWithin_le_of_mem hu vu, fun h ↦ (f_an h).mono v'u, f',
      fun y hy ↦ (hf' y (v'u hy)).mono v'u, hv⟩
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
      n : WithTop ENat
      hn : Ne n ↑Top.top
      ⊢ (∀ (x : E), Membership.mem s x → Exists fun u => And (Membership.mem (nhdsWi …
    -/
  · intro h x hx
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
      n : WithTop ENat
      hn : Ne n ↑Top.top
      h : ∀ (x : E), Membership.mem s x → Exists fun u => And (Membership.mem (nhdsW …
      x : E
      hx : Membership.mem s x
      ⊢ ContDiffWithinAt 𝕜 (HAdd.hAdd n 1) f s x
    -/
    rw [contDiffWithinAt_succ_iff_hasFDerivWithinAt hn]
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
      n : WithTop ENat
      hn : Ne n ↑Top.top
      h : ∀ (x : E), Membership.mem s x → Exists fun u => And (Membership.mem (nhdsW …
      x : E
      hx : Membership.mem s x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    rcases h x hx with ⟨u, u_nhbd, f_an, f', hu, hf'⟩
    /-
      case mpr.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h : ∀ (x : E), Membership.mem s x → Exists fun u => And (Membership.mem (nhdsW …
      x : E
      hx : Membership.mem s x
      u : Set E
      u_nhbd : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      hf' : ContDiffOn 𝕜 n f' u
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    have : x ∈ u := mem_of_mem_nhdsWithin (mem_insert _ _) u_nhbd
    /-
      case mpr.intro.intro.intro.intro.intro
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
      hn : Ne n ↑Top.top
      h : ∀ (x : E), Membership.mem s x → Exists fun u => And (Membership.mem (nhdsW …
      x : E
      hx : Membership.mem s x
      u : Set E
      u_nhbd : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      f_an : Eq n Top.top → AnalyticOn 𝕜 f u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      hu : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      hf' : ContDiffOn 𝕜 n f' u
      this : Membership.mem u x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (A …
    -/
    exact ⟨u, u_nhbd, f_an, f', hu, hf' x this⟩
    /-
      🎉 no goals
    -/



@[simp]
theorem contDiffOn_zero : ContDiffOn 𝕜 0 f s ↔ ContinuousOn f s := by
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
    ⊢ Iff (ContDiffOn 𝕜 0 f s) (ContinuousOn f s)
  -/
  refine ⟨fun H => H.continuousOn, fun H => fun x hx m hm ↦ ?_⟩
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
    H : ContinuousOn f s
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) 0
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
  -/
  have : (m : WithTop ℕ∞) = 0 := le_antisymm (mod_cast hm) bot_le
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
    H : ContinuousOn f s
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) 0
    this : Eq (↑m) 0
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
  -/
  rw [this]
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
    H : ContinuousOn f s
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) 0
    this : Eq (↑m) 0
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
  -/
  refine ⟨insert x s, self_mem_nhdsWithin, ftaylorSeriesWithin 𝕜 f s, ?_⟩
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
    H : ContinuousOn f s
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) 0
    this : Eq (↑m) 0
    ⊢ HasFTaylorSeriesUpToOn 0 f (ftaylorSeriesWithin 𝕜 f s) (Insert.insert x s)
  -/
  rw [hasFTaylorSeriesUpToOn_zero_iff]
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
    H : ContinuousOn f s
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) 0
    this : Eq (↑m) 0
    ⊢ And (ContinuousOn f (Insert.insert x s)) (∀ (x_1 : E), Membership.mem (Inser …
  -/
  exact ⟨by rwa [insert_eq_of_mem hx], fun x _ => by simp [ftaylorSeriesWithin]⟩
  /-
    🎉 no goals
  -/


theorem contDiffWithinAt_zero (hx : x ∈ s) :
    ContDiffWithinAt 𝕜 0 f s x ↔ ∃ u ∈ 𝓝[s] x, ContinuousOn f (s ∩ u) := by
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
    hx : Membership.mem s x
    ⊢ Iff (ContDiffWithinAt 𝕜 0 f s x) (Exists fun u => And (Membership.mem (nhdsW …
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
      x : E
      hx : Membership.mem s x
      ⊢ ContDiffWithinAt 𝕜 0 f s x → Exists fun u => And (Membership.mem (nhdsWithin …
    -/
  · intro h
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
      x : E
      hx : Membership.mem s x
      h : ContDiffWithinAt 𝕜 0 f s x
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (ContinuousOn f (Int …
    -/
    obtain ⟨u, H, p, hp⟩ := h 0 le_rfl
    /-
      case mp.intro.intro.intro
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
      hx : Membership.mem s x
      h : ContDiffWithinAt 𝕜 0 f s x
      u : Set E
      H : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      hp : HasFTaylorSeriesUpToOn (↑0) f p u
      ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (ContinuousOn f (Int …
    -/
    refine ⟨u, ?_, ?_⟩
      /-
        case mp.intro.intro.intro.refine_1
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
        hx : Membership.mem s x
        h : ContDiffWithinAt 𝕜 0 f s x
        u : Set E
        H : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        hp : HasFTaylorSeriesUpToOn (↑0) f p u
        ⊢ Membership.mem (nhdsWithin x s) u
      -/
    · simpa [hx] using H
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.intro.intro.refine_2
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
        hx : Membership.mem s x
        h : ContDiffWithinAt 𝕜 0 f s x
        u : Set E
        H : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        hp : HasFTaylorSeriesUpToOn (↑0) f p u
        ⊢ ContinuousOn f (Inter.inter s u)
      -/
    · simp only [Nat.cast_zero, hasFTaylorSeriesUpToOn_zero_iff] at hp
      /-
        case mp.intro.intro.intro.refine_2
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
        hx : Membership.mem s x
        h : ContDiffWithinAt 𝕜 0 f s x
        u : Set E
        H : Membership.mem (nhdsWithin x (Insert.insert x s)) u
        p : E → FormalMultilinearSeries 𝕜 E F
        hp : And (ContinuousOn f u) (∀ (x : E), Membership.mem u x → Eq (p x 0).curry0 …
        ⊢ ContinuousOn f (Inter.inter s u)
      -/
      exact hp.1.mono inter_subset_right
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
      x : E
      hx : Membership.mem s x
      ⊢ (Exists fun u => And (Membership.mem (nhdsWithin x s) u) (ContinuousOn f (In …
    -/
  · rintro ⟨u, H, hu⟩
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
      x : E
      hx : Membership.mem s x
      u : Set E
      H : Membership.mem (nhdsWithin x s) u
      hu : ContinuousOn f (Inter.inter s u)
      ⊢ ContDiffWithinAt 𝕜 0 f s x
    -/
    rw [← contDiffWithinAt_inter' H]
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
      x : E
      hx : Membership.mem s x
      u : Set E
      H : Membership.mem (nhdsWithin x s) u
      hu : ContinuousOn f (Inter.inter s u)
      ⊢ ContDiffWithinAt 𝕜 0 f (Inter.inter s u) x
    -/
    have h' : x ∈ s ∩ u := ⟨hx, mem_of_mem_nhdsWithin hx H⟩
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
      x : E
      hx : Membership.mem s x
      u : Set E
      H : Membership.mem (nhdsWithin x s) u
      hu : ContinuousOn f (Inter.inter s u)
      h' : Membership.mem (Inter.inter s u) x
      ⊢ ContDiffWithinAt 𝕜 0 f (Inter.inter s u) x
    -/
    exact (contDiffOn_zero.mpr hu).contDiffWithinAt h'
    /-
      🎉 no goals
    -/


/-- When a function is `C^n` in a set `s` of unique differentiability, it admits
`ftaylorSeriesWithin 𝕜 f s` as a Taylor series up to order `n` in `s`. -/
protected theorem ContDiffOn.ftaylorSeriesWithin
    (h : ContDiffOn 𝕜 n f s) (hs : UniqueDiffOn 𝕜 s) :
    HasFTaylorSeriesUpToOn n f (ftaylorSeriesWithin 𝕜 f s) s := by
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
    h : ContDiffOn 𝕜 n f s
    hs : UniqueDiffOn 𝕜 s
    ⊢ HasFTaylorSeriesUpToOn n f (ftaylorSeriesWithin 𝕜 f s) s
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
      n : WithTop ENat
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      ⊢ ∀ (x : E), Membership.mem s x → Eq (ftaylorSeriesWithin 𝕜 f s x 0).curry0 (f …
    -/
  · intro x _
    simp only [ftaylorSeriesWithin, ContinuousMultilinearMap.curry0_apply,
      iteratedFDerivWithin_zero_apply]
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
      n : WithTop ENat
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      ⊢ ∀ (m : Nat), LT.lt (↑m) n → ∀ (x : E), Membership.mem s x → HasFDerivWithinA …
    -/
  · intro m hm x hx
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
      n : WithTop ENat
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) (ftaylorSeriesWit …
    -/
    have : (m + 1 : ℕ) ≤ n := ENat.add_one_natCast_le_withTop_of_lt hm
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
      n : WithTop ENat
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      this : LE.le (↑(HAdd.hAdd m 1)) n
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) (ftaylorSeriesWit …
    -/
    rcases (h x hx).of_le this _ le_rfl with ⟨u, hu, p, Hp⟩
    /-
      case fderivWithin.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      this : LE.le (↑(HAdd.hAdd m 1)) n
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd m 1)) f p u
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) (ftaylorSeriesWit …
    -/
    rw [insert_eq_of_mem hx] at hu
    /-
      case fderivWithin.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      this : LE.le (↑(HAdd.hAdd m 1)) n
      u : Set E
      hu : Membership.mem (nhdsWithin x s) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd m 1)) f p u
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) (ftaylorSeriesWit …
    -/
    rcases mem_nhdsWithin.1 hu with ⟨o, o_open, xo, ho⟩
    /-
      case fderivWithin.intro.intro.intro.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      this : LE.le (↑(HAdd.hAdd m 1)) n
      u : Set E
      hu : Membership.mem (nhdsWithin x s) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd m 1)) f p u
      o : Set E
      o_open : IsOpen o
      xo : Membership.mem o x
      ho : HasSubset.Subset (Inter.inter o s) u
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) (ftaylorSeriesWit …
    -/
    rw [inter_comm] at ho
    have : p x m.succ = ftaylorSeriesWithin 𝕜 f s x m.succ := by
      change p x m.succ = iteratedFDerivWithin 𝕜 m.succ f s x
      rw [← iteratedFDerivWithin_inter_open o_open xo]
      exact (Hp.mono ho).eq_iteratedFDerivWithin_of_uniqueDiffOn le_rfl (hs.inter o_open) ⟨hx, xo⟩
    /-
      case fderivWithin.intro.intro.intro.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LT.lt (↑m) n
      x : E
      hx : Membership.mem s x
      this✝ : LE.le (↑(HAdd.hAdd m 1)) n
      u : Set E
      hu : Membership.mem (nhdsWithin x s) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑(HAdd.hAdd m 1)) f p u
      o : Set E
      o_open : IsOpen o
      xo : Membership.mem o x
      ho : HasSubset.Subset (Inter.inter s o) u
      this : Eq (p x m.succ) (ftaylorSeriesWithin 𝕜 f s x m.succ)
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x m) (ftaylorSeriesWit …
    -/
    rw [← this, ← hasFDerivWithinAt_inter (IsOpen.mem_nhds o_open xo)]
    have A : ∀ y ∈ s ∩ o, p y m = ftaylorSeriesWithin 𝕜 f s y m := by
      rintro y ⟨hy, yo⟩
      change p y m = iteratedFDerivWithin 𝕜 m f s y
      rw [← iteratedFDerivWithin_inter_open o_open yo]
      exact
        (Hp.mono ho).eq_iteratedFDerivWithin_of_uniqueDiffOn (mod_cast Nat.le_succ m)
          (hs.inter o_open) ⟨hy, yo⟩
    exact
      ((Hp.mono ho).fderivWithin m (mod_cast lt_add_one m) x ⟨hx, xo⟩).congr
        (fun y hy => (A y hy).symm) (A x ⟨hx, xo⟩).symm
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
      n : WithTop ENat
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      ⊢ ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => ftaylorSeriesWithin 𝕜 f s …
    -/
  · intro m hm
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
      n : WithTop ENat
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LE.le (↑m) n
      ⊢ ContinuousOn (fun x => ftaylorSeriesWithin 𝕜 f s x m) s
    -/
    apply continuousOn_of_locally_continuousOn
    /-
      case cont.h
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LE.le (↑m) n
      ⊢ ∀ (x : E), Membership.mem s x → Exists fun t => And (IsOpen t) (And (Members …
    -/
    intro x hx
    /-
      case cont.h
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      ⊢ Exists fun t => And (IsOpen t) (And (Membership.mem t x) (ContinuousOn (fun  …
    -/
    rcases (h x hx).of_le hm _ le_rfl with ⟨u, hu, p, Hp⟩
    /-
      case cont.h.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑m) f p u
      ⊢ Exists fun t => And (IsOpen t) (And (Membership.mem t x) (ContinuousOn (fun  …
    -/
    rcases mem_nhdsWithin.1 hu with ⟨o, o_open, xo, ho⟩
    /-
      case cont.h.intro.intro.intro.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑m) f p u
      o : Set E
      o_open : IsOpen o
      xo : Membership.mem o x
      ho : HasSubset.Subset (Inter.inter o (Insert.insert x s)) u
      ⊢ Exists fun t => And (IsOpen t) (And (Membership.mem t x) (ContinuousOn (fun  …
    -/
    rw [insert_eq_of_mem hx] at ho
    /-
      case cont.h.intro.intro.intro.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑m) f p u
      o : Set E
      o_open : IsOpen o
      xo : Membership.mem o x
      ho : HasSubset.Subset (Inter.inter o s) u
      ⊢ Exists fun t => And (IsOpen t) (And (Membership.mem t x) (ContinuousOn (fun  …
    -/
    rw [inter_comm] at ho
    /-
      case cont.h.intro.intro.intro.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑m) f p u
      o : Set E
      o_open : IsOpen o
      xo : Membership.mem o x
      ho : HasSubset.Subset (Inter.inter s o) u
      ⊢ Exists fun t => And (IsOpen t) (And (Membership.mem t x) (ContinuousOn (fun  …
    -/
    refine ⟨o, o_open, xo, ?_⟩
    have A : ∀ y ∈ s ∩ o, p y m = ftaylorSeriesWithin 𝕜 f s y m := by
      rintro y ⟨hy, yo⟩
      change p y m = iteratedFDerivWithin 𝕜 m f s y
      rw [← iteratedFDerivWithin_inter_open o_open yo]
      exact (Hp.mono ho).eq_iteratedFDerivWithin_of_uniqueDiffOn le_rfl (hs.inter o_open) ⟨hy, yo⟩
    /-
      case cont.h.intro.intro.intro.intro.intro.intro
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
      h : ContDiffOn 𝕜 n f s
      hs : UniqueDiffOn 𝕜 s
      m : Nat
      hm : LE.le (↑m) n
      x : E
      hx : Membership.mem s x
      u : Set E
      hu : Membership.mem (nhdsWithin x (Insert.insert x s)) u
      p : E → FormalMultilinearSeries 𝕜 E F
      Hp : HasFTaylorSeriesUpToOn (↑m) f p u
      o : Set E
      o_open : IsOpen o
      xo : Membership.mem o x
      ho : HasSubset.Subset (Inter.inter s o) u
      A : ∀ (y : E), Membership.mem (Inter.inter s o) y → Eq (p y m) (ftaylorSeriesW …
      ⊢ ContinuousOn (fun x => ftaylorSeriesWithin 𝕜 f s x m) (Inter.inter s o)
    -/
    exact ((Hp.mono ho).cont m le_rfl).congr fun y hy => (A y hy).symm
    /-
      🎉 no goals
    -/


theorem iteratedFDerivWithin_subset {n : ℕ} (st : s ⊆ t) (hs : UniqueDiffOn 𝕜 s)
    (ht : UniqueDiffOn 𝕜 t) (h : ContDiffOn 𝕜 n f t) (hx : x ∈ s) :
    iteratedFDerivWithin 𝕜 n f s x = iteratedFDerivWithin 𝕜 n f t x :=
  (((h.ftaylorSeriesWithin ht).mono st).eq_iteratedFDerivWithin_of_uniqueDiffOn le_rfl hs hx).symm


/-- On a set with unique differentiability, an analytic function is automatically `C^ω`, as its
successive derivatives are also analytic. This does not require completeness of the space. See
also `AnalyticOn.contDiffOn_of_completeSpace`.-/
theorem AnalyticOn.contDiffOn (h : AnalyticOn 𝕜 f s) (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 n f s := by
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
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    ⊢ ContDiffOn 𝕜 n f s
  -/
  suffices ContDiffOn 𝕜 ω f s from this.of_le le_top
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
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    ⊢ ContDiffOn 𝕜 Top.top f s
  -/
  rcases h.exists_hasFTaylorSeriesUpToOn hs with ⟨p, hp⟩
  /-
    case intro
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
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    p : E → FormalMultilinearSeries 𝕜 E F
    hp : And (HasFTaylorSeriesUpToOn Top.top f p s) (∀ (i : Nat), AnalyticOn 𝕜 (fu …
    ⊢ ContDiffOn 𝕜 Top.top f s
  -/
  intro x hx
  /-
    case intro
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
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    p : E → FormalMultilinearSeries 𝕜 E F
    hp : And (HasFTaylorSeriesUpToOn Top.top f p s) (∀ (i : Nat), AnalyticOn 𝕜 (fu …
    x : E
    hx : Membership.mem s x
    ⊢ ContDiffWithinAt 𝕜 Top.top f s x
  -/
  refine ⟨s, ?_, p, hp⟩
  /-
    case intro
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
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    p : E → FormalMultilinearSeries 𝕜 E F
    hp : And (HasFTaylorSeriesUpToOn Top.top f p s) (∀ (i : Nat), AnalyticOn 𝕜 (fu …
    x : E
    hx : Membership.mem s x
    ⊢ Membership.mem (nhdsWithin x (Insert.insert x s)) s
  -/
  rw [insert_eq_of_mem hx]
  /-
    case intro
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
    h : AnalyticOn 𝕜 f s
    hs : UniqueDiffOn 𝕜 s
    p : E → FormalMultilinearSeries 𝕜 E F
    hp : And (HasFTaylorSeriesUpToOn Top.top f p s) (∀ (i : Nat), AnalyticOn 𝕜 (fu …
    x : E
    hx : Membership.mem s x
    ⊢ Membership.mem (nhdsWithin x s) s
  -/
  exact self_mem_nhdsWithin
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-26")]
alias AnalyticWithinOn.contDiffOn := AnalyticOn.contDiffOn


/-- On a set with unique differentiability, an analytic function is automatically `C^ω`, as its
successive derivatives are also analytic. This does not require completeness of the space. See
also `AnalyticOnNhd.contDiffOn_of_completeSpace`. -/
theorem AnalyticOnNhd.contDiffOn (h : AnalyticOnNhd 𝕜 f s) (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 n f s := h.analyticOn.contDiffOn hs


/-- An analytic function is automatically `C^ω` in a complete space -/
theorem AnalyticOn.contDiffOn_of_completeSpace [CompleteSpace F] (h : AnalyticOn 𝕜 f s) :
    ContDiffOn 𝕜 n f s :=
  fun x hx ↦ (h x hx).contDiffWithinAt


/-- An analytic function is automatically `C^ω` in a complete space -/
theorem AnalyticOnNhd.contDiffOn_of_completeSpace [CompleteSpace F] (h : AnalyticOnNhd 𝕜 f s) :
    ContDiffOn 𝕜 n f s :=
  h.analyticOn.contDiffOn_of_completeSpace


theorem contDiffOn_of_continuousOn_differentiableOn {n : ℕ∞}
    (Hcont : ∀ m : ℕ, m ≤ n → ContinuousOn (fun x => iteratedFDerivWithin 𝕜 m f s x) s)
    (Hdiff : ∀ m : ℕ, m < n →
      DifferentiableOn 𝕜 (fun x => iteratedFDerivWithin 𝕜 m f s x) s) :
    ContDiffOn 𝕜 n f s := by
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
    n : ENat
    Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
    Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
    ⊢ ContDiffOn 𝕜 (↑n) f s
  -/
  intro x hx m hm
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
    n : ENat
    Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
    Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) n
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x s)) u) (E …
  -/
  rw [insert_eq_of_mem hx]
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
    n : ENat
    Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
    Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) n
    ⊢ Exists fun u => And (Membership.mem (nhdsWithin x s) u) (Exists fun p => Has …
  -/
  refine ⟨s, self_mem_nhdsWithin, ftaylorSeriesWithin 𝕜 f s, ?_⟩
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
    n : ENat
    Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
    Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
    x : E
    hx : Membership.mem s x
    m : Nat
    hm : LE.le (↑m) n
    ⊢ HasFTaylorSeriesUpToOn (↑m) f (ftaylorSeriesWithin 𝕜 f s) s
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
      n : ENat
      Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
      Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
      x : E
      hx : Membership.mem s x
      m : Nat
      hm : LE.le (↑m) n
      ⊢ ∀ (x : E), Membership.mem s x → Eq (ftaylorSeriesWithin 𝕜 f s x 0).curry0 (f …
    -/
  · intro y _
    simp only [ftaylorSeriesWithin, ContinuousMultilinearMap.curry0_apply,
      iteratedFDerivWithin_zero_apply]
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
      n : ENat
      Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
      Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
      x : E
      hx : Membership.mem s x
      m : Nat
      hm : LE.le (↑m) n
      ⊢ ∀ (m_1 : Nat), LT.lt ↑m_1 ↑m → ∀ (x : E), Membership.mem s x → HasFDerivWith …
    -/
  · intro k hk y hy
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
      n : ENat
      Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
      Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
      x : E
      hx : Membership.mem s x
      m : Nat
      hm : LE.le (↑m) n
      k : Nat
      hk : LT.lt ↑k ↑m
      y : E
      hy : Membership.mem s y
      ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x k) (ftaylorSeriesWit …
    -/
    convert (Hdiff k (lt_of_lt_of_le (mod_cast hk) (mod_cast hm)) y hy).hasFDerivWithinAt
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
      n : ENat
      Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
      Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
      x : E
      hx : Membership.mem s x
      m : Nat
      hm : LE.le (↑m) n
      ⊢ ∀ (m_1 : Nat), LE.le ↑m_1 ↑m → ContinuousOn (fun x => ftaylorSeriesWithin 𝕜  …
    -/
  · intro k hk
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
      n : ENat
      Hcont : ∀ (m : Nat), LE.le (↑m) n → ContinuousOn (fun x => iteratedFDerivWithi …
      Hdiff : ∀ (m : Nat), LT.lt (↑m) n → DifferentiableOn 𝕜 (fun x => iteratedFDeri …
      x : E
      hx : Membership.mem s x
      m : Nat
      hm : LE.le (↑m) n
      k : Nat
      hk : LE.le ↑k ↑m
      ⊢ ContinuousOn (fun x => ftaylorSeriesWithin 𝕜 f s x k) s
    -/
    exact Hcont k (le_trans (mod_cast hk) (mod_cast hm))
    /-
      🎉 no goals
    -/


theorem contDiffOn_of_differentiableOn {n : ℕ∞}
    (h : ∀ m : ℕ, m ≤ n → DifferentiableOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s) :
    ContDiffOn 𝕜 n f s :=
  contDiffOn_of_continuousOn_differentiableOn (fun m hm => (h m hm).continuousOn) fun m hm =>
    h m (le_of_lt hm)


theorem contDiffOn_of_analyticOn_iteratedFDerivWithin
    (h : ∀ m, AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s) :
    ContDiffOn 𝕜 n f s := by
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
    h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
    ⊢ ContDiffOn 𝕜 n f s
  -/
  suffices ContDiffOn 𝕜 ω f s from this.of_le le_top
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
    h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
    ⊢ ContDiffOn 𝕜 Top.top f s
  -/
  intro x hx
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
    h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
    x : E
    hx : Membership.mem s x
    ⊢ ContDiffWithinAt 𝕜 Top.top f s x
  -/
  refine ⟨insert x s, self_mem_nhdsWithin, ftaylorSeriesWithin 𝕜 f s, ?_, ?_⟩
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
      n : WithTop ENat
      h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
      x : E
      hx : Membership.mem s x
      ⊢ HasFTaylorSeriesUpToOn Top.top f (ftaylorSeriesWithin 𝕜 f s) (Insert.insert  …
    -/
  · rw [insert_eq_of_mem hx]
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
      n : WithTop ENat
      h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
      x : E
      hx : Membership.mem s x
      ⊢ HasFTaylorSeriesUpToOn Top.top f (ftaylorSeriesWithin 𝕜 f s) s
    -/
    constructor
      /-
        case refine_1.zero_eq
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
        h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
        x : E
        hx : Membership.mem s x
        ⊢ ∀ (x : E), Membership.mem s x → Eq (ftaylorSeriesWithin 𝕜 f s x 0).curry0 (f …
      -/
    · intro y _
      simp only [ftaylorSeriesWithin, ContinuousMultilinearMap.curry0_apply,
        iteratedFDerivWithin_zero_apply]
      /-
        case refine_1.fderivWithin
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
        h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
        x : E
        hx : Membership.mem s x
        ⊢ ∀ (m : Nat), LT.lt (↑m) Top.top → ∀ (x : E), Membership.mem s x → HasFDerivW …
      -/
    · intro k _ y hy
      /-
        case refine_1.fderivWithin
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
        h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
        x : E
        hx : Membership.mem s x
        k : Nat
        a✝ : LT.lt (↑k) Top.top
        y : E
        hy : Membership.mem s y
        ⊢ HasFDerivWithinAt (fun x => ftaylorSeriesWithin 𝕜 f s x k) (ftaylorSeriesWit …
      -/
      exact ((h k).differentiableOn y hy).hasFDerivWithinAt
      /-
        🎉 no goals
      -/
      /-
        case refine_1.cont
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
        h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
        x : E
        hx : Membership.mem s x
        ⊢ ∀ (m : Nat), LE.le (↑m) Top.top → ContinuousOn (fun x => ftaylorSeriesWithin …
      -/
    · intro k _
      /-
        case refine_1.cont
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
        h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
        x : E
        hx : Membership.mem s x
        k : Nat
        a✝ : LE.le (↑k) Top.top
        ⊢ ContinuousOn (fun x => ftaylorSeriesWithin 𝕜 f s x k) s
      -/
      exact (h k).continuousOn
      /-
        🎉 no goals
      -/
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
      n : WithTop ENat
      h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
      x : E
      hx : Membership.mem s x
      ⊢ ∀ (i : Nat), AnalyticOn 𝕜 (fun x => ftaylorSeriesWithin 𝕜 f s x i) (Insert.i …
    -/
  · intro i
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
      n : WithTop ENat
      h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
      x : E
      hx : Membership.mem s x
      i : Nat
      ⊢ AnalyticOn 𝕜 (fun x => ftaylorSeriesWithin 𝕜 f s x i) (Insert.insert x s)
    -/
    rw [insert_eq_of_mem hx]
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
      n : WithTop ENat
      h : ∀ (m : Nat), AnalyticOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
      x : E
      hx : Membership.mem s x
      i : Nat
      ⊢ AnalyticOn 𝕜 (fun x => ftaylorSeriesWithin 𝕜 f s x i) s
    -/
    exact h i
    /-
      🎉 no goals
    -/


theorem contDiffOn_omega_iff_analyticOn (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 ω f s ↔ AnalyticOn 𝕜 f s :=
  ⟨fun h m ↦ h.analyticOn m, fun h ↦ h.contDiffOn hs⟩


theorem ContDiffOn.continuousOn_iteratedFDerivWithin {m : ℕ} (h : ContDiffOn 𝕜 n f s)
    (hmn : m ≤ n) (hs : UniqueDiffOn 𝕜 s) : ContinuousOn (iteratedFDerivWithin 𝕜 m f s) s :=
  ((h.of_le hmn).ftaylorSeriesWithin hs).cont m le_rfl


theorem ContDiffOn.differentiableOn_iteratedFDerivWithin {m : ℕ} (h : ContDiffOn 𝕜 n f s)
    (hmn : m < n) (hs : UniqueDiffOn 𝕜 s) :
    DifferentiableOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s := by
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
    m : Nat
    h : ContDiffOn 𝕜 n f s
    hmn : LT.lt (↑m) n
    hs : UniqueDiffOn 𝕜 s
    ⊢ DifferentiableOn 𝕜 (iteratedFDerivWithin 𝕜 m f s) s
  -/
  intro x hx
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
    m : Nat
    h : ContDiffOn 𝕜 n f s
    hmn : LT.lt (↑m) n
    hs : UniqueDiffOn 𝕜 s
    x : E
    hx : Membership.mem s x
    ⊢ DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f s) s x
  -/
  have : (m + 1 : ℕ) ≤ n := ENat.add_one_natCast_le_withTop_of_lt hmn
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
    m : Nat
    h : ContDiffOn 𝕜 n f s
    hmn : LT.lt (↑m) n
    hs : UniqueDiffOn 𝕜 s
    x : E
    hx : Membership.mem s x
    this : LE.le (↑(HAdd.hAdd m 1)) n
    ⊢ DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f s) s x
  -/
  apply (((h.of_le this).ftaylorSeriesWithin hs).fderivWithin m ?_ x hx).differentiableWithinAt
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
    m : Nat
    h : ContDiffOn 𝕜 n f s
    hmn : LT.lt (↑m) n
    hs : UniqueDiffOn 𝕜 s
    x : E
    hx : Membership.mem s x
    this : LE.le (↑(HAdd.hAdd m 1)) n
    ⊢ LT.lt ↑m ↑(HAdd.hAdd m 1)
  -/
  exact_mod_cast lt_add_one m
  /-
    🎉 no goals
  -/


theorem ContDiffWithinAt.differentiableWithinAt_iteratedFDerivWithin {m : ℕ}
    (h : ContDiffWithinAt 𝕜 n f s x) (hmn : m < n) (hs : UniqueDiffOn 𝕜 (insert x s)) :
    DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f s) s x := by
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
    m : Nat
    h : ContDiffWithinAt 𝕜 n f s x
    hmn : LT.lt (↑m) n
    hs : UniqueDiffOn 𝕜 (Insert.insert x s)
    ⊢ DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f s) s x
  -/
  have : (m + 1 : WithTop ℕ∞) ≠ ∞ := Ne.symm (ne_of_beq_false rfl)
  rcases h.contDiffOn' (ENat.add_one_natCast_le_withTop_of_lt hmn) (by simp [this])
    with ⟨u, uo, xu, hu⟩
  /-
    case intro.intro.intro
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
    m : Nat
    h : ContDiffWithinAt 𝕜 n f s x
    hmn : LT.lt (↑m) n
    hs : UniqueDiffOn 𝕜 (Insert.insert x s)
    this : Ne (HAdd.hAdd (↑m) 1) ↑Top.top
    u : Set E
    uo : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 (↑(HAdd.hAdd m 1)) f (Inter.inter (Insert.insert x s) u)
    ⊢ DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f s) s x
  -/
  set t := insert x s ∩ u
  have A : t =ᶠ[𝓝[≠] x] s := by
    simp only [set_eventuallyEq_iff_inf_principal, ← nhdsWithin_inter']
    rw [← inter_assoc, nhdsWithin_inter_of_mem', ← diff_eq_compl_inter, insert_diff_of_mem,
      diff_eq_compl_inter]
    exacts [rfl, mem_nhdsWithin_of_mem_nhds (uo.mem_nhds xu)]
  have B : iteratedFDerivWithin 𝕜 m f s =ᶠ[𝓝 x] iteratedFDerivWithin 𝕜 m f t :=
    iteratedFDerivWithin_eventually_congr_set' _ A.symm _
  have C : DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f t) t x :=
    hu.differentiableOn_iteratedFDerivWithin (Nat.cast_lt.2 m.lt_succ_self) (hs.inter uo) x
      ⟨mem_insert _ _, xu⟩
  /-
    case intro.intro.intro
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
    m : Nat
    h : ContDiffWithinAt 𝕜 n f s x
    hmn : LT.lt (↑m) n
    hs : UniqueDiffOn 𝕜 (Insert.insert x s)
    this : Ne (HAdd.hAdd (↑m) 1) ↑Top.top
    u : Set E
    uo : IsOpen u
    xu : Membership.mem u x
    t : Set E := Inter.inter (Insert.insert x s) u
    hu : ContDiffOn 𝕜 (↑(HAdd.hAdd m 1)) f t
    A : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).EventuallyEq t s
    B : (nhds x).EventuallyEq (iteratedFDerivWithin 𝕜 m f s) (iteratedFDerivWithin …
    C : DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f t) t x
    ⊢ DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f s) s x
  -/
  rw [differentiableWithinAt_congr_set' _ A] at C
  /-
    case intro.intro.intro
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
    m : Nat
    h : ContDiffWithinAt 𝕜 n f s x
    hmn : LT.lt (↑m) n
    hs : UniqueDiffOn 𝕜 (Insert.insert x s)
    this : Ne (HAdd.hAdd (↑m) 1) ↑Top.top
    u : Set E
    uo : IsOpen u
    xu : Membership.mem u x
    t : Set E := Inter.inter (Insert.insert x s) u
    hu : ContDiffOn 𝕜 (↑(HAdd.hAdd m 1)) f t
    A : (nhdsWithin x (HasCompl.compl (Singleton.singleton x))).EventuallyEq t s
    B : (nhds x).EventuallyEq (iteratedFDerivWithin 𝕜 m f s) (iteratedFDerivWithin …
    C : DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f t) s x
    ⊢ DifferentiableWithinAt 𝕜 (iteratedFDerivWithin 𝕜 m f s) s x
  -/
  exact C.congr_of_eventuallyEq (B.filter_mono inf_le_left) B.self_of_nhds
  /-
    🎉 no goals
  -/


theorem contDiffOn_iff_continuousOn_differentiableOn {n : ℕ∞} (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 n f s ↔
      (∀ m : ℕ, m ≤ n → ContinuousOn (fun x => iteratedFDerivWithin 𝕜 m f s x) s) ∧
        ∀ m : ℕ, m < n → DifferentiableOn 𝕜 (fun x => iteratedFDerivWithin 𝕜 m f s x) s :=
  ⟨fun h => ⟨fun _m hm => h.continuousOn_iteratedFDerivWithin (mod_cast hm) hs,
      fun _m hm => h.differentiableOn_iteratedFDerivWithin (mod_cast hm) hs⟩,
    fun h => contDiffOn_of_continuousOn_differentiableOn h.1 h.2⟩


theorem contDiffOn_nat_iff_continuousOn_differentiableOn {n : ℕ} (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 n f s ↔
      (∀ m : ℕ, m ≤ n → ContinuousOn (fun x => iteratedFDerivWithin 𝕜 m f s x) s) ∧
        ∀ m : ℕ, m < n → DifferentiableOn 𝕜 (fun x => iteratedFDerivWithin 𝕜 m f s x) s := by
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
    hs : UniqueDiffOn 𝕜 s
    ⊢ Iff (ContDiffOn 𝕜 (↑n) f s) (And (∀ (m : Nat), LE.le m n → ContinuousOn (fun …
  -/
  rw [show n = ((n : ℕ∞) : WithTop ℕ∞) from rfl, contDiffOn_iff_continuousOn_differentiableOn hs]
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
    hs : UniqueDiffOn 𝕜 s
    ⊢ Iff (And (∀ (m : Nat), LE.le ↑m ↑n → ContinuousOn (fun x => iteratedFDerivWi …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem contDiffOn_succ_of_fderivWithin (hf : DifferentiableOn 𝕜 f s)
    (h' : n = ω → AnalyticOn 𝕜 f s)
    (h : ContDiffOn 𝕜 n (fun y => fderivWithin 𝕜 f s y) s) : ContDiffOn 𝕜 (n + 1) f s := by
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
    hf : DifferentiableOn 𝕜 f s
    h' : Eq n Top.top → AnalyticOn 𝕜 f s
    h : ContDiffOn 𝕜 n (fun y => fderivWithin 𝕜 f s y) s
    ⊢ ContDiffOn 𝕜 (HAdd.hAdd n 1) f s
  -/
  rcases eq_or_ne n ∞ with rfl | hn
    /-
      case inl
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
      hf : DifferentiableOn 𝕜 f s
      h' : Eq (↑Top.top) Top.top → AnalyticOn 𝕜 f s
      h : ContDiffOn 𝕜 (↑Top.top) (fun y => fderivWithin 𝕜 f s y) s
      ⊢ ContDiffOn 𝕜 (HAdd.hAdd (↑Top.top) 1) f s
    -/
  · rw [ENat.coe_top_add_one, contDiffOn_infty]
    /-
      case inl
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
      hf : DifferentiableOn 𝕜 f s
      h' : Eq (↑Top.top) Top.top → AnalyticOn 𝕜 f s
      h : ContDiffOn 𝕜 (↑Top.top) (fun y => fderivWithin 𝕜 f s y) s
      ⊢ ∀ (n : Nat), ContDiffOn 𝕜 (↑n) f s
    -/
    intro m x hx
    /-
      case inl
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
      hf : DifferentiableOn 𝕜 f s
      h' : Eq (↑Top.top) Top.top → AnalyticOn 𝕜 f s
      h : ContDiffOn 𝕜 (↑Top.top) (fun y => fderivWithin 𝕜 f s y) s
      m : Nat
      x : E
      hx : Membership.mem s x
      ⊢ ContDiffWithinAt 𝕜 (↑m) f s x
    -/
    apply ContDiffWithinAt.of_le _ (show (m : WithTop ℕ∞) ≤ m + 1 from le_self_add)
    rw [contDiffWithinAt_succ_iff_hasFDerivWithinAt (by simp),
      insert_eq_of_mem hx]
    exact ⟨s, self_mem_nhdsWithin, (by simp), fderivWithin 𝕜 f s,
      fun y hy => (hf y hy).hasFDerivWithinAt, (h x hx).of_le (mod_cast le_top)⟩
    /-
      case inr
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
      hf : DifferentiableOn 𝕜 f s
      h' : Eq n Top.top → AnalyticOn 𝕜 f s
      h : ContDiffOn 𝕜 n (fun y => fderivWithin 𝕜 f s y) s
      hn : Ne n ↑Top.top
      ⊢ ContDiffOn 𝕜 (HAdd.hAdd n 1) f s
    -/
  · intro x hx
    rw [contDiffWithinAt_succ_iff_hasFDerivWithinAt hn,
      insert_eq_of_mem hx]
    exact ⟨s, self_mem_nhdsWithin, h', fderivWithin 𝕜 f s,
      fun y hy => (hf y hy).hasFDerivWithinAt, h x hx⟩


theorem contDiffOn_of_analyticOn_of_fderivWithin (hf : AnalyticOn 𝕜 f s)
    (h : ContDiffOn 𝕜 ω (fun y ↦ fderivWithin 𝕜 f s y) s) : ContDiffOn 𝕜 n f s := by
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
    hf : AnalyticOn 𝕜 f s
    h : ContDiffOn 𝕜 Top.top (fun y => fderivWithin 𝕜 f s y) s
    ⊢ ContDiffOn 𝕜 n f s
  -/
  suffices ContDiffOn 𝕜 (ω + 1) f s from this.of_le le_top
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
    hf : AnalyticOn 𝕜 f s
    h : ContDiffOn 𝕜 Top.top (fun y => fderivWithin 𝕜 f s y) s
    ⊢ ContDiffOn 𝕜 (HAdd.hAdd Top.top 1) f s
  -/
  exact contDiffOn_succ_of_fderivWithin hf.differentiableOn (fun _ ↦ hf) h
  /-
    🎉 no goals
  -/


/-- A function is `C^(n + 1)` on a domain with unique derivatives if and only if it is
differentiable there, and its derivative (expressed with `fderivWithin`) is `C^n`. -/
theorem contDiffOn_succ_iff_fderivWithin (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 (n + 1) f s ↔
      DifferentiableOn 𝕜 f s ∧ (n = ω → AnalyticOn 𝕜 f s) ∧
      ContDiffOn 𝕜 n (fderivWithin 𝕜 f s) s := by
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
    hs : UniqueDiffOn 𝕜 s
    ⊢ Iff (ContDiffOn 𝕜 (HAdd.hAdd n 1) f s) (And (DifferentiableOn 𝕜 f s) (And (E …
  -/
  refine ⟨fun H => ?_, fun h => contDiffOn_succ_of_fderivWithin h.1 h.2.1 h.2.2⟩
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
    hs : UniqueDiffOn 𝕜 s
    H : ContDiffOn 𝕜 (HAdd.hAdd n 1) f s
    ⊢ And (DifferentiableOn 𝕜 f s) (And (Eq n Top.top → AnalyticOn 𝕜 f s) (ContDif …
  -/
  refine ⟨H.differentiableOn le_add_self, ?_, fun x hx => ?_⟩
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
      n : WithTop ENat
      hs : UniqueDiffOn 𝕜 s
      H : ContDiffOn 𝕜 (HAdd.hAdd n 1) f s
      ⊢ Eq n Top.top → AnalyticOn 𝕜 f s
    -/
  · rintro rfl
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
      hs : UniqueDiffOn 𝕜 s
      H : ContDiffOn 𝕜 (HAdd.hAdd Top.top 1) f s
      ⊢ AnalyticOn 𝕜 f s
    -/
    exact H.analyticOn
    /-
      🎉 no goals
    -/
  have A (m : ℕ) (hm : m ≤ n) : ContDiffWithinAt 𝕜 m (fun y => fderivWithin 𝕜 f s y) s x := by
    rcases (contDiffWithinAt_succ_iff_hasFDerivWithinAt (n := m) (ne_of_beq_false rfl)).1
      (H.of_le (add_le_add_right hm 1) x hx) with ⟨u, hu, -, f', hff', hf'⟩
    rcases mem_nhdsWithin.1 hu with ⟨o, o_open, xo, ho⟩
    rw [inter_comm, insert_eq_of_mem hx] at ho
    have := hf'.mono ho
    rw [contDiffWithinAt_inter' (mem_nhdsWithin_of_mem_nhds (IsOpen.mem_nhds o_open xo))] at this
    apply this.congr_of_eventuallyEq_of_mem _ hx
    have : o ∩ s ∈ 𝓝[s] x := mem_nhdsWithin.2 ⟨o, o_open, xo, Subset.refl _⟩
    rw [inter_comm] at this
    refine Filter.eventuallyEq_of_mem this fun y hy => ?_
    have A : fderivWithin 𝕜 f (s ∩ o) y = f' y :=
      ((hff' y (ho hy)).mono ho).fderivWithin (hs.inter o_open y hy)
    rwa [fderivWithin_inter (o_open.mem_nhds hy.2)] at A
  match n with
  | ω => exact (H.analyticOn.fderivWithin hs).contDiffOn hs (n := ω) x hx
  | ∞ => exact contDiffWithinAt_infty.2 (fun m ↦ A m (mod_cast le_top))
  | (n : ℕ) => exact A n le_rfl


theorem contDiffOn_succ_iff_hasFDerivWithinAt_of_uniqueDiffOn (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 (n + 1) f s ↔ (n = ω → AnalyticOn 𝕜 f s) ∧
      ∃ f' : E → E →L[𝕜] F, ContDiffOn 𝕜 n f' s ∧ ∀ x, x ∈ s → HasFDerivWithinAt f (f' x) s x := by
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
    hs : UniqueDiffOn 𝕜 s
    ⊢ Iff (ContDiffOn 𝕜 (HAdd.hAdd n 1) f s) (And (Eq n Top.top → AnalyticOn 𝕜 f s …
  -/
  rw [contDiffOn_succ_iff_fderivWithin hs]
  refine ⟨fun h => ⟨h.2.1, fderivWithin 𝕜 f s, h.2.2,
    fun x hx => (h.1 x hx).hasFDerivWithinAt⟩, fun ⟨f_an, h⟩ => ?_⟩
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
    hs : UniqueDiffOn 𝕜 s
    x✝ : And (Eq n Top.top → AnalyticOn 𝕜 f s) (Exists fun f' => And (ContDiffOn 𝕜 …
    f_an : Eq n Top.top → AnalyticOn 𝕜 f s
    h : Exists fun f' => And (ContDiffOn 𝕜 n f' s) (∀ (x : E), Membership.mem s x  …
    ⊢ And (DifferentiableOn 𝕜 f s) (And (Eq n Top.top → AnalyticOn 𝕜 f s) (ContDif …
  -/
  rcases h with ⟨f', h1, h2⟩
  /-
    case intro.intro
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
    hs : UniqueDiffOn 𝕜 s
    x✝ : And (Eq n Top.top → AnalyticOn 𝕜 f s) (Exists fun f' => And (ContDiffOn 𝕜 …
    f_an : Eq n Top.top → AnalyticOn 𝕜 f s
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    h1 : ContDiffOn 𝕜 n f' s
    h2 : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    ⊢ And (DifferentiableOn 𝕜 f s) (And (Eq n Top.top → AnalyticOn 𝕜 f s) (ContDif …
  -/
  refine ⟨fun x hx => (h2 x hx).differentiableWithinAt, f_an, fun x hx => ?_⟩
  /-
    case intro.intro
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
    hs : UniqueDiffOn 𝕜 s
    x✝ : And (Eq n Top.top → AnalyticOn 𝕜 f s) (Exists fun f' => And (ContDiffOn 𝕜 …
    f_an : Eq n Top.top → AnalyticOn 𝕜 f s
    f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
    h1 : ContDiffOn 𝕜 n f' s
    h2 : ∀ (x : E), Membership.mem s x → HasFDerivWithinAt f (f' x) s x
    x : E
    hx : Membership.mem s x
    ⊢ ContDiffWithinAt 𝕜 n (fderivWithin 𝕜 f s) s x
  -/
  exact (h1 x hx).congr_of_mem (fun y hy => (h2 y hy).fderivWithin (hs y hy)) hx
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-27")]
alias contDiffOn_succ_iff_hasFDerivWithin := contDiffOn_succ_iff_hasFDerivWithinAt_of_uniqueDiffOn


theorem contDiffOn_infty_iff_fderivWithin (hs : UniqueDiffOn 𝕜 s) :
    ContDiffOn 𝕜 ∞ f s ↔ DifferentiableOn 𝕜 f s ∧ ContDiffOn 𝕜 ∞ (fderivWithin 𝕜 f s) s := by
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
    hs : UniqueDiffOn 𝕜 s
    ⊢ Iff (ContDiffOn 𝕜 (↑Top.top) f s) (And (DifferentiableOn 𝕜 f s) (ContDiffOn  …
  -/
  rw [show ∞ = ∞ + 1 from rfl, contDiffOn_succ_iff_fderivWithin hs]
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
    hs : UniqueDiffOn 𝕜 s
    ⊢ Iff (And (DifferentiableOn 𝕜 f s) (And (Eq (↑Top.top) Top.top → AnalyticOn 𝕜 …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-27")]
alias contDiffOn_top_iff_fderivWithin := contDiffOn_infty_iff_fderivWithin


/-- A function is `C^(n + 1)` on an open domain if and only if it is
differentiable there, and its derivative (expressed with `fderiv`) is `C^n`. -/
theorem contDiffOn_succ_iff_fderiv_of_isOpen (hs : IsOpen s) :
    ContDiffOn 𝕜 (n + 1) f s ↔
      DifferentiableOn 𝕜 f s ∧ (n = ω → AnalyticOn 𝕜 f s) ∧
      ContDiffOn 𝕜 n (fderiv 𝕜 f) s := by
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
    hs : IsOpen s
    ⊢ Iff (ContDiffOn 𝕜 (HAdd.hAdd n 1) f s) (And (DifferentiableOn 𝕜 f s) (And (E …
  -/
  rw [contDiffOn_succ_iff_fderivWithin hs.uniqueDiffOn]
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
    hs : IsOpen s
    ⊢ Iff (And (DifferentiableOn 𝕜 f s) (And (Eq n Top.top → AnalyticOn 𝕜 f s) (Co …
  -/
  exact Iff.rfl.and (Iff.rfl.and (contDiffOn_congr fun x hx ↦ fderivWithin_of_isOpen hs hx))
  /-
    🎉 no goals
  -/


theorem contDiffOn_infty_iff_fderiv_of_isOpen (hs : IsOpen s) :
    ContDiffOn 𝕜 ∞ f s ↔ DifferentiableOn 𝕜 f s ∧ ContDiffOn 𝕜 ∞ (fderiv 𝕜 f) s := by
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
    hs : IsOpen s
    ⊢ Iff (ContDiffOn 𝕜 (↑Top.top) f s) (And (DifferentiableOn 𝕜 f s) (ContDiffOn  …
  -/
  rw [show ∞ = ∞ + 1 from rfl, contDiffOn_succ_iff_fderiv_of_isOpen hs]
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
    hs : IsOpen s
    ⊢ Iff (And (DifferentiableOn 𝕜 f s) (And (Eq (↑Top.top) Top.top → AnalyticOn 𝕜 …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-27")]
alias contDiffOn_top_iff_fderiv_of_isOpen := contDiffOn_infty_iff_fderiv_of_isOpen


protected theorem ContDiffOn.fderivWithin (hf : ContDiffOn 𝕜 n f s) (hs : UniqueDiffOn 𝕜 s)
    (hmn : m + 1 ≤ n) : ContDiffOn 𝕜 m (fderivWithin 𝕜 f s) s :=
  ((contDiffOn_succ_iff_fderivWithin hs).1 (hf.of_le hmn)).2.2


theorem ContDiffOn.fderiv_of_isOpen (hf : ContDiffOn 𝕜 n f s) (hs : IsOpen s) (hmn : m + 1 ≤ n) :
    ContDiffOn 𝕜 m (fderiv 𝕜 f) s :=
  (hf.fderivWithin hs.uniqueDiffOn hmn).congr fun _ hx => (fderivWithin_of_isOpen hs hx).symm


theorem ContDiffOn.continuousOn_fderivWithin (h : ContDiffOn 𝕜 n f s) (hs : UniqueDiffOn 𝕜 s)
    (hn : 1 ≤ n) : ContinuousOn (fderivWithin 𝕜 f s) s :=
  ((contDiffOn_succ_iff_fderivWithin hs).1
    (h.of_le (show 0 + (1 : WithTop ℕ∞) ≤ n from hn))).2.2.continuousOn


theorem ContDiffOn.continuousOn_fderiv_of_isOpen (h : ContDiffOn 𝕜 n f s) (hs : IsOpen s)
    (hn : 1 ≤ n) : ContinuousOn (fderiv 𝕜 f) s :=
  ((contDiffOn_succ_iff_fderiv_of_isOpen hs).1
    (h.of_le (show 0 + (1 : WithTop ℕ∞) ≤ n from hn))).2.2.continuousOn


variable (𝕜) in
/-- A function is continuously differentiable up to `n` at a point `x` if, for any integer `k ≤ n`,
there is a neighborhood of `x` where `f` admits derivatives up to order `n`, which are continuous.
-/
def ContDiffAt (n : WithTop ℕ∞) (f : E → F) (x : E) : Prop :=
  ContDiffWithinAt 𝕜 n f univ x


theorem contDiffWithinAt_univ : ContDiffWithinAt 𝕜 n f univ x ↔ ContDiffAt 𝕜 n f x :=
  Iff.rfl


theorem contDiffAt_infty : ContDiffAt 𝕜 ∞ f x ↔ ∀ n : ℕ, ContDiffAt 𝕜 n f x := by
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
    ⊢ Iff (ContDiffAt 𝕜 (↑Top.top) f x) (∀ (n : Nat), ContDiffAt 𝕜 (↑n) f x)
  -/
  simp [← contDiffWithinAt_univ, contDiffWithinAt_infty]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-27")] alias contDiffAt_top := contDiffAt_infty


theorem ContDiffAt.contDiffWithinAt (h : ContDiffAt 𝕜 n f x) : ContDiffWithinAt 𝕜 n f s x :=
  h.mono (subset_univ _)


theorem ContDiffWithinAt.contDiffAt (h : ContDiffWithinAt 𝕜 n f s x) (hx : s ∈ 𝓝 x) :
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
                               h : ContDiffWithinAt 𝕜 n f s x
                               hx : Membership.mem (nhds x) s
                               ⊢ ContDiffAt 𝕜 n f x
                             -/
    ContDiffAt 𝕜 n f x := by rwa [ContDiffAt, ← contDiffWithinAt_inter hx, univ_inter]
                             /-
                               🎉 no goals
                             -/


theorem contDiffWithinAt_iff_contDiffAt (h : s ∈ 𝓝 x) :
    ContDiffWithinAt 𝕜 n f s x ↔ ContDiffAt 𝕜 n f x := by
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
    h : Membership.mem (nhds x) s
    ⊢ Iff (ContDiffWithinAt 𝕜 n f s x) (ContDiffAt 𝕜 n f x)
  -/
  rw [← univ_inter s, contDiffWithinAt_inter h, contDiffWithinAt_univ]
  /-
    🎉 no goals
  -/


theorem IsOpen.contDiffOn_iff (hs : IsOpen s) :
    ContDiffOn 𝕜 n f s ↔ ∀ ⦃a⦄, a ∈ s → ContDiffAt 𝕜 n f a :=
  forall₂_congr fun _ => contDiffWithinAt_iff_contDiffAt ∘ hs.mem_nhds


theorem ContDiffOn.contDiffAt (h : ContDiffOn 𝕜 n f s) (hx : s ∈ 𝓝 x) :
    ContDiffAt 𝕜 n f x :=
  (h _ (mem_of_mem_nhds hx)).contDiffAt hx


theorem ContDiffAt.congr_of_eventuallyEq (h : ContDiffAt 𝕜 n f x) (hg : f₁ =ᶠ[𝓝 x] f) :
    ContDiffAt 𝕜 n f₁ x :=
                                     /-
                                       𝕜 : Type u
                                       inst✝⁴ : NontriviallyNormedField 𝕜
                                       E : Type uE
                                       inst✝³ : NormedAddCommGroup E
                                       inst✝² : NormedSpace 𝕜 E
                                       F : Type uF
                                       inst✝¹ : NormedAddCommGroup F
                                       inst✝ : NormedSpace 𝕜 F
                                       f f₁ : E → F
                                       x : E
                                       n : WithTop ENat
                                       h : ContDiffAt 𝕜 n f x
                                       hg : (nhds x).EventuallyEq f₁ f
                                       ⊢ (nhdsWithin x Set.univ).EventuallyEq f₁ f
                                     -/
  h.congr_of_eventuallyEq_of_mem (by rwa [nhdsWithin_univ]) (mem_univ x)
                                     /-
                                       🎉 no goals
                                     -/


theorem ContDiffAt.of_le (h : ContDiffAt 𝕜 n f x) (hmn : m ≤ n) : ContDiffAt 𝕜 m f x :=
  ContDiffWithinAt.of_le h hmn


theorem ContDiffAt.continuousAt (h : ContDiffAt 𝕜 n f x) : ContinuousAt f x := by
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
    n : WithTop ENat
    h : ContDiffAt 𝕜 n f x
    ⊢ ContinuousAt f x
  -/
  simpa [continuousWithinAt_univ] using h.continuousWithinAt
  /-
    🎉 no goals
  -/


theorem ContDiffAt.analyticAt (h : ContDiffAt 𝕜 ω f x) : AnalyticAt 𝕜 f x := by
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
    h : ContDiffAt 𝕜 Top.top f x
    ⊢ AnalyticAt 𝕜 f x
  -/
  rw [← contDiffWithinAt_univ] at h
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
    h : ContDiffWithinAt 𝕜 Top.top f Set.univ x
    ⊢ AnalyticAt 𝕜 f x
  -/
  rw [← analyticWithinAt_univ]
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
    h : ContDiffWithinAt 𝕜 Top.top f Set.univ x
    ⊢ AnalyticWithinAt 𝕜 f Set.univ x
  -/
  exact h.analyticWithinAt
  /-
    🎉 no goals
  -/


/-- In a complete space, a function which is analytic at a point is also `C^ω` there.
Note that the same statement for `AnalyticOn` does not require completeness, see
`AnalyticOn.contDiffOn`. -/
theorem AnalyticAt.contDiffAt [CompleteSpace F] (h : AnalyticAt 𝕜 f x) :
    ContDiffAt 𝕜 n f x := by
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type uF
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : WithTop ENat
    inst✝ : CompleteSpace F
    h : AnalyticAt 𝕜 f x
    ⊢ ContDiffAt 𝕜 n f x
  -/
  rw [← contDiffWithinAt_univ]
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type uF
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : WithTop ENat
    inst✝ : CompleteSpace F
    h : AnalyticAt 𝕜 f x
    ⊢ ContDiffWithinAt 𝕜 n f Set.univ x
  -/
  rw [← analyticWithinAt_univ] at h
  /-
    𝕜 : Type u
    inst✝⁵ : NontriviallyNormedField 𝕜
    E : Type uE
    inst✝⁴ : NormedAddCommGroup E
    inst✝³ : NormedSpace 𝕜 E
    F : Type uF
    inst✝² : NormedAddCommGroup F
    inst✝¹ : NormedSpace 𝕜 F
    f : E → F
    x : E
    n : WithTop ENat
    inst✝ : CompleteSpace F
    h : AnalyticWithinAt 𝕜 f Set.univ x
    ⊢ ContDiffWithinAt 𝕜 n f Set.univ x
  -/
  exact h.contDiffWithinAt
  /-
    🎉 no goals
  -/


@[simp]
theorem contDiffWithinAt_compl_self :
    ContDiffWithinAt 𝕜 n f {x}ᶜ x ↔ ContDiffAt 𝕜 n f x := by
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
    n : WithTop ENat
    ⊢ Iff (ContDiffWithinAt 𝕜 n f (HasCompl.compl (Singleton.singleton x)) x) (Con …
  -/
  rw [compl_eq_univ_diff, contDiffWithinAt_diff_singleton, contDiffWithinAt_univ]
  /-
    🎉 no goals
  -/


/-- If a function is `C^n` with `n ≥ 1` at a point, then it is differentiable there. -/
theorem ContDiffAt.differentiableAt (h : ContDiffAt 𝕜 n f x) (hn : 1 ≤ n) :
    DifferentiableAt 𝕜 f x := by
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
    n : WithTop ENat
    h : ContDiffAt 𝕜 n f x
    hn : LE.le 1 n
    ⊢ DifferentiableAt 𝕜 f x
  -/
  simpa [hn, differentiableWithinAt_univ] using h.differentiableWithinAt
  /-
    🎉 no goals
  -/


nonrec lemma ContDiffAt.contDiffOn (h : ContDiffAt 𝕜 n f x) (hm : m ≤ n) (h' : m = ∞ → n = ω):
    ∃ u ∈ 𝓝 x, ContDiffOn 𝕜 m f u := by
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
    m n : WithTop ENat
    h : ContDiffAt 𝕜 n f x
    hm : LE.le m n
    h' : Eq m ↑Top.top → Eq n Top.top
    ⊢ Exists fun u => And (Membership.mem (nhds x) u) (ContDiffOn 𝕜 m f u)
  -/
  simpa [nhdsWithin_univ] using h.contDiffOn hm h'
  /-
    🎉 no goals
  -/


/-- A function is `C^(n + 1)` at a point iff locally, it has a derivative which is `C^n`. -/
theorem contDiffAt_succ_iff_hasFDerivAt {n : ℕ} :
    ContDiffAt 𝕜 (n + 1) f x ↔ ∃ f' : E → E →L[𝕜] F,
      (∃ u ∈ 𝓝 x, ∀ x ∈ u, HasFDerivAt f (f' x) x) ∧ ContDiffAt 𝕜 n f' x := by
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
    ⊢ Iff (ContDiffAt 𝕜 (HAdd.hAdd (↑n) 1) f x) (Exists fun f' => And (Exists fun  …
  -/
  rw [← contDiffWithinAt_univ, contDiffWithinAt_succ_iff_hasFDerivWithinAt (by simp)]
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
    ⊢ Iff (Exists fun u => And (Membership.mem (nhdsWithin x (Insert.insert x Set. …
  -/
  simp only [nhdsWithin_univ, exists_prop, mem_univ, insert_eq_of_mem]
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
    ⊢ Iff (Exists fun u => And (Membership.mem (nhds x) u) (And (Eq (↑n) Top.top → …
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
      x : E
      n : Nat
      ⊢ (Exists fun u => And (Membership.mem (nhds x) u) (And (Eq (↑n) Top.top → Ana …
    -/
  · rintro ⟨u, H, -, f', h_fderiv, h_cont_diff⟩
    /-
      case mp.intro.intro.intro.intro.intro
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
      u : Set E
      H : Membership.mem (nhds x) u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      h_fderiv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      h_cont_diff : ContDiffWithinAt 𝕜 (↑n) f' u x
      ⊢ Exists fun f' => And (Exists fun u => And (Membership.mem (nhds x) u) (∀ (x  …
    -/
    rcases mem_nhds_iff.mp H with ⟨t, htu, ht, hxt⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
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
      u : Set E
      H : Membership.mem (nhds x) u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      h_fderiv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      h_cont_diff : ContDiffWithinAt 𝕜 (↑n) f' u x
      t : Set E
      htu : HasSubset.Subset t u
      ht : IsOpen t
      hxt : Membership.mem t x
      ⊢ Exists fun f' => And (Exists fun u => And (Membership.mem (nhds x) u) (∀ (x  …
    -/
    refine ⟨f', ⟨t, ?_⟩, h_cont_diff.contDiffAt H⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
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
      u : Set E
      H : Membership.mem (nhds x) u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      h_fderiv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      h_cont_diff : ContDiffWithinAt 𝕜 (↑n) f' u x
      t : Set E
      htu : HasSubset.Subset t u
      ht : IsOpen t
      hxt : Membership.mem t x
      ⊢ And (Membership.mem (nhds x) t) (∀ (x : E), Membership.mem t x → HasFDerivAt …
    -/
    refine ⟨mem_nhds_iff.mpr ⟨t, Subset.rfl, ht, hxt⟩, ?_⟩
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
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
      u : Set E
      H : Membership.mem (nhds x) u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      h_fderiv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      h_cont_diff : ContDiffWithinAt 𝕜 (↑n) f' u x
      t : Set E
      htu : HasSubset.Subset t u
      ht : IsOpen t
      hxt : Membership.mem t x
      ⊢ ∀ (x : E), Membership.mem t x → HasFDerivAt f (f' x) x
    -/
    intro y hyt
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
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
      u : Set E
      H : Membership.mem (nhds x) u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      h_fderiv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      h_cont_diff : ContDiffWithinAt 𝕜 (↑n) f' u x
      t : Set E
      htu : HasSubset.Subset t u
      ht : IsOpen t
      hxt : Membership.mem t x
      y : E
      hyt : Membership.mem t y
      ⊢ HasFDerivAt f (f' y) y
    -/
    refine (h_fderiv y (htu hyt)).hasFDerivAt ?_
    /-
      case mp.intro.intro.intro.intro.intro.intro.intro.intro
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
      u : Set E
      H : Membership.mem (nhds x) u
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      h_fderiv : ∀ (x : E), Membership.mem u x → HasFDerivWithinAt f (f' x) u x
      h_cont_diff : ContDiffWithinAt 𝕜 (↑n) f' u x
      t : Set E
      htu : HasSubset.Subset t u
      ht : IsOpen t
      hxt : Membership.mem t x
      y : E
      hyt : Membership.mem t y
      ⊢ Membership.mem (nhds y) u
    -/
    exact mem_nhds_iff.mpr ⟨t, htu, ht, hyt⟩
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
      x : E
      n : Nat
      ⊢ (Exists fun f' => And (Exists fun u => And (Membership.mem (nhds x) u) (∀ (x …
    -/
  · rintro ⟨f', ⟨u, H, h_fderiv⟩, h_cont_diff⟩
    /-
      case mpr.intro.intro.intro.intro
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
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      h_cont_diff : ContDiffAt 𝕜 (↑n) f' x
      u : Set E
      H : Membership.mem (nhds x) u
      h_fderiv : ∀ (x : E), Membership.mem u x → HasFDerivAt f (f' x) x
      ⊢ Exists fun u => And (Membership.mem (nhds x) u) (And (Eq (↑n) Top.top → Anal …
    -/
    refine ⟨u, H, by simp, f', fun x hxu ↦ ?_, h_cont_diff.contDiffWithinAt⟩
    /-
      case mpr.intro.intro.intro.intro
      𝕜 : Type u
      inst✝⁴ : NontriviallyNormedField 𝕜
      E : Type uE
      inst✝³ : NormedAddCommGroup E
      inst✝² : NormedSpace 𝕜 E
      F : Type uF
      inst✝¹ : NormedAddCommGroup F
      inst✝ : NormedSpace 𝕜 F
      f : E → F
      x✝ : E
      n : Nat
      f' : E → ContinuousLinearMap (RingHom.id 𝕜) E F
      h_cont_diff : ContDiffAt 𝕜 (↑n) f' x✝
      u : Set E
      H : Membership.mem (nhds x✝) u
      h_fderiv : ∀ (x : E), Membership.mem u x → HasFDerivAt f (f' x) x
      x : E
      hxu : Membership.mem u x
      ⊢ HasFDerivWithinAt f (f' x) u x
    -/
    exact (h_fderiv x hxu).hasFDerivWithinAt
    /-
      🎉 no goals
    -/


protected theorem ContDiffAt.eventually (h : ContDiffAt 𝕜 n f x) (h' : n ≠ ∞) :
    ∀ᶠ y in 𝓝 x, ContDiffAt 𝕜 n f y := by
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
    n : WithTop ENat
    h : ContDiffAt 𝕜 n f x
    h' : Ne n ↑Top.top
    ⊢ Filter.Eventually (fun y => ContDiffAt 𝕜 n f y) (nhds x)
  -/
  simpa [nhdsWithin_univ] using ContDiffWithinAt.eventually h h'
  /-
    🎉 no goals
  -/


theorem iteratedFDerivWithin_eq_iteratedFDeriv {n : ℕ}
    (hs : UniqueDiffOn 𝕜 s) (h : ContDiffAt 𝕜 n f x) (hx : x ∈ s) :
    iteratedFDerivWithin 𝕜 n f s x = iteratedFDeriv 𝕜 n f x := by
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
    h : ContDiffAt 𝕜 (↑n) f x
    hx : Membership.mem s x
    ⊢ Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDeriv 𝕜 n f x)
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
    s : Set E
    f : E → F
    x : E
    n : Nat
    hs : UniqueDiffOn 𝕜 s
    h : ContDiffAt 𝕜 (↑n) f x
    hx : Membership.mem s x
    ⊢ Eq (iteratedFDerivWithin 𝕜 n f s x) (iteratedFDerivWithin 𝕜 n f Set.univ x)
  -/
  rcases h.contDiffOn' le_rfl (by simp) with ⟨u, u_open, xu, hu⟩
  rw [← iteratedFDerivWithin_inter_open u_open xu,
    ← iteratedFDerivWithin_inter_open u_open xu (s := univ)]
  /-
    case intro.intro.intro
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
    h : ContDiffAt 𝕜 (↑n) f x
    hx : Membership.mem s x
    u : Set E
    u_open : IsOpen u
    xu : Membership.mem u x
    hu : ContDiffOn 𝕜 (↑n) f (Inter.inter (Insert.insert x Set.univ) u)
    ⊢ Eq (iteratedFDerivWithin 𝕜 n f (Inter.inter s u) x) (iteratedFDerivWithin 𝕜  …
  -/
  apply iteratedFDerivWithin_subset
    /-
      case intro.intro.intro.st
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
      h : ContDiffAt 𝕜 (↑n) f x
      hx : Membership.mem s x
      u : Set E
      u_open : IsOpen u
      xu : Membership.mem u x
      hu : ContDiffOn 𝕜 (↑n) f (Inter.inter (Insert.insert x Set.univ) u)
      ⊢ HasSubset.Subset (Inter.inter s u) (Inter.inter Set.univ u)
    -/
  · exact inter_subset_inter_left _ (subset_univ _)
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.hs
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
      h : ContDiffAt 𝕜 (↑n) f x
      hx : Membership.mem s x
      u : Set E
      u_open : IsOpen u
      xu : Membership.mem u x
      hu : ContDiffOn 𝕜 (↑n) f (Inter.inter (Insert.insert x Set.univ) u)
      ⊢ UniqueDiffOn 𝕜 (Inter.inter s u)
    -/
  · exact hs.inter u_open
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.ht
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
      h : ContDiffAt 𝕜 (↑n) f x
      hx : Membership.mem s x
      u : Set E
      u_open : IsOpen u
      xu : Membership.mem u x
      hu : ContDiffOn 𝕜 (↑n) f (Inter.inter (Insert.insert x Set.univ) u)
      ⊢ UniqueDiffOn 𝕜 (Inter.inter Set.univ u)
    -/
  · apply uniqueDiffOn_univ.inter u_open
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.h
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
      h : ContDiffAt 𝕜 (↑n) f x
      hx : Membership.mem s x
      u : Set E
      u_open : IsOpen u
      xu : Membership.mem u x
      hu : ContDiffOn 𝕜 (↑n) f (Inter.inter (Insert.insert x Set.univ) u)
      ⊢ ContDiffOn 𝕜 (↑n) f (Inter.inter Set.univ u)
    -/
  · simpa using hu
    /-
      🎉 no goals
    -/
    /-
      case intro.intro.intro.hx
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
      h : ContDiffAt 𝕜 (↑n) f x
      hx : Membership.mem s x
      u : Set E
      u_open : IsOpen u
      xu : Membership.mem u x
      hu : ContDiffOn 𝕜 (↑n) f (Inter.inter (Insert.insert x Set.univ) u)
      ⊢ Membership.mem (Inter.inter s u) x
    -/
  · exact ⟨hx, xu⟩
    /-
      🎉 no goals
    -/


variable (𝕜) in
/-- A function is continuously differentiable up to `n` if it admits derivatives up to
order `n`, which are continuous. Contrary to the case of definitions in domains (where derivatives
might not be unique) we do not need to localize the definition in space or time.
-/
def ContDiff (n : WithTop ℕ∞) (f : E → F) : Prop :=
  match n with
  | ω => ∃ p : E → FormalMultilinearSeries 𝕜 E F, HasFTaylorSeriesUpTo ⊤ f p
      ∧ ∀ i, AnalyticOnNhd 𝕜 (fun x ↦ p x i) univ
  | (n : ℕ∞) => ∃ p : E → FormalMultilinearSeries 𝕜 E F, HasFTaylorSeriesUpTo n f p


/-- If `f` has a Taylor series up to `n`, then it is `C^n`. -/
theorem HasFTaylorSeriesUpTo.contDiff {n : ℕ∞} {f' : E → FormalMultilinearSeries 𝕜 E F}
    (hf : HasFTaylorSeriesUpTo n f f') : ContDiff 𝕜 n f :=
  ⟨f', hf⟩


theorem contDiffOn_univ : ContDiffOn 𝕜 n f univ ↔ ContDiff 𝕜 n f := by
  match n with
  | ω =>
    constructor
    · intro H
      use ftaylorSeriesWithin 𝕜 f univ
      rw [← hasFTaylorSeriesUpToOn_univ_iff]
      refine ⟨H.ftaylorSeriesWithin uniqueDiffOn_univ, fun i ↦ ?_⟩
      rw [← analyticOn_univ]
      exact H.analyticOn.iteratedFDerivWithin uniqueDiffOn_univ _
    · rintro ⟨p, hp, h'p⟩ x _
      exact ⟨univ, Filter.univ_sets _, p, (hp.hasFTaylorSeriesUpToOn univ).of_le le_top,
        fun i ↦ (h'p i).analyticOn⟩
  | (n : ℕ∞) =>
    constructor
    · intro H
      use ftaylorSeriesWithin 𝕜 f univ
      rw [← hasFTaylorSeriesUpToOn_univ_iff]
      exact H.ftaylorSeriesWithin uniqueDiffOn_univ
    · rintro ⟨p, hp⟩ x _ m hm
      exact ⟨univ, Filter.univ_sets _, p,
        (hp.hasFTaylorSeriesUpToOn univ).of_le (mod_cast hm)⟩


theorem contDiff_iff_contDiffAt : ContDiff 𝕜 n f ↔ ∀ x, ContDiffAt 𝕜 n f x := by
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
    ⊢ Iff (ContDiff 𝕜 n f) (∀ (x : E), ContDiffAt 𝕜 n f x)
  -/
  simp [← contDiffOn_univ, ContDiffOn, ContDiffAt]
  /-
    🎉 no goals
  -/


theorem ContDiff.contDiffAt (h : ContDiff 𝕜 n f) : ContDiffAt 𝕜 n f x :=
  contDiff_iff_contDiffAt.1 h x


theorem ContDiff.contDiffWithinAt (h : ContDiff 𝕜 n f) : ContDiffWithinAt 𝕜 n f s x :=
  h.contDiffAt.contDiffWithinAt


theorem contDiff_infty : ContDiff 𝕜 ∞ f ↔ ∀ n : ℕ, ContDiff 𝕜 n f := by
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
    ⊢ Iff (ContDiff 𝕜 (↑Top.top) f) (∀ (n : Nat), ContDiff 𝕜 (↑n) f)
  -/
  simp [contDiffOn_univ.symm, contDiffOn_infty]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-25")] alias contDiff_top := contDiff_infty


@[deprecated (since := "2024-11-25")] alias contDiff_infty_iff_contDiff_omega := contDiff_infty


theorem contDiff_all_iff_nat : (∀ n : ℕ∞, ContDiff 𝕜 n f) ↔ ∀ n : ℕ, ContDiff 𝕜 n f := by
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
    ⊢ Iff (∀ (n : ENat), ContDiff 𝕜 (↑n) f) (∀ (n : Nat), ContDiff 𝕜 (↑n) f)
  -/
  simp only [← contDiffOn_univ, contDiffOn_all_iff_nat]
  /-
    🎉 no goals
  -/


theorem ContDiff.contDiffOn (h : ContDiff 𝕜 n f) : ContDiffOn 𝕜 n f s :=
  (contDiffOn_univ.2 h).mono (subset_univ _)


@[simp]
theorem contDiff_zero : ContDiff 𝕜 0 f ↔ Continuous f := by
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
    ⊢ Iff (ContDiff 𝕜 0 f) (Continuous f)
  -/
  rw [← contDiffOn_univ, continuous_iff_continuousOn_univ]
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
    ⊢ Iff (ContDiffOn 𝕜 0 f Set.univ) (ContinuousOn f Set.univ)
  -/
  exact contDiffOn_zero
  /-
    🎉 no goals
  -/


theorem contDiffAt_zero : ContDiffAt 𝕜 0 f x ↔ ∃ u ∈ 𝓝 x, ContinuousOn f u := by
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
    ⊢ Iff (ContDiffAt 𝕜 0 f x) (Exists fun u => And (Membership.mem (nhds x) u) (C …
  -/
  rw [← contDiffWithinAt_univ]; simp [contDiffWithinAt_zero, nhdsWithin_univ]
                                /-
                                  🎉 no goals
                                -/


theorem contDiffAt_one_iff :
    ContDiffAt 𝕜 1 f x ↔
      ∃ f' : E → E →L[𝕜] F, ∃ u ∈ 𝓝 x, ContinuousOn f' u ∧ ∀ x ∈ u, HasFDerivAt f (f' x) x := by
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
    ⊢ Iff (ContDiffAt 𝕜 1 f x) (Exists fun f' => Exists fun u => And (Membership.m …
  -/
  rw [show (1 : WithTop ℕ∞) = (0 : ℕ) + 1 from rfl]
  simp_rw [contDiffAt_succ_iff_hasFDerivAt, show ((0 : ℕ) : WithTop ℕ∞) = 0 from rfl,
    contDiffAt_zero, exists_mem_and_iff antitone_bforall antitone_continuousOn, and_comm]


theorem ContDiff.of_le (h : ContDiff 𝕜 n f) (hmn : m ≤ n) : ContDiff 𝕜 m f :=
  contDiffOn_univ.1 <| (contDiffOn_univ.2 h).of_le hmn


theorem ContDiff.of_succ (h : ContDiff 𝕜 (n + 1) f) : ContDiff 𝕜 n f :=
  h.of_le le_self_add


theorem ContDiff.one_of_succ (h : ContDiff 𝕜 (n + 1) f) : ContDiff 𝕜 1 f := by
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
    h : ContDiff 𝕜 (HAdd.hAdd n 1) f
    ⊢ ContDiff 𝕜 1 f
  -/
  apply h.of_le le_add_self
  /-
    🎉 no goals
  -/


theorem ContDiff.continuous (h : ContDiff 𝕜 n f) : Continuous f :=
  contDiff_zero.1 (h.of_le bot_le)


/-- If a function is `C^n` with `n ≥ 1`, then it is differentiable. -/
theorem ContDiff.differentiable (h : ContDiff 𝕜 n f) (hn : 1 ≤ n) : Differentiable 𝕜 f :=
  differentiableOn_univ.1 <| (contDiffOn_univ.2 h).differentiableOn hn


theorem contDiff_iff_forall_nat_le {n : ℕ∞} :
    ContDiff 𝕜 n f ↔ ∀ m : ℕ, ↑m ≤ n → ContDiff 𝕜 m f := by
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
    n : ENat
    ⊢ Iff (ContDiff 𝕜 (↑n) f) (∀ (m : Nat), LE.le (↑m) n → ContDiff 𝕜 (↑m) f)
  -/
  simp_rw [← contDiffOn_univ]; exact contDiffOn_iff_forall_nat_le
                               /-
                                 🎉 no goals
                               -/


/-- A function is `C^(n+1)` iff it has a `C^n` derivative. -/
theorem contDiff_succ_iff_hasFDerivAt {n : ℕ} :
    ContDiff 𝕜 (n + 1) f ↔
      ∃ f' : E → E →L[𝕜] F, ContDiff 𝕜 n f' ∧ ∀ x, HasFDerivAt f (f' x) x := by
  simp only [← contDiffOn_univ, ← hasFDerivWithinAt_univ, Set.mem_univ, forall_true_left,
    contDiffOn_succ_iff_hasFDerivWithinAt_of_uniqueDiffOn uniqueDiffOn_univ,
    WithTop.natCast_ne_top, analyticOn_univ, false_implies, true_and]


theorem contDiff_one_iff_hasFDerivAt : ContDiff 𝕜 1 f ↔
    ∃ f' : E → E →L[𝕜] F, Continuous f' ∧ ∀ x, HasFDerivAt f (f' x) x := by
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
    ⊢ Iff (ContDiff 𝕜 1 f) (Exists fun f' => And (Continuous f') (∀ (x : E), HasFD …
  -/
  convert contDiff_succ_iff_hasFDerivAt using 4; simp
                                                 /-
                                                   🎉 no goals
                                                 -/


theorem AnalyticOn.contDiff (hf : AnalyticOn 𝕜 f univ) : ContDiff 𝕜 n f := by
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
    hf : AnalyticOn 𝕜 f Set.univ
    ⊢ ContDiff 𝕜 n f
  -/
  rw [← contDiffOn_univ]
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
    hf : AnalyticOn 𝕜 f Set.univ
    ⊢ ContDiffOn 𝕜 n f Set.univ
  -/
  exact hf.contDiffOn (n := n) uniqueDiffOn_univ
  /-
    🎉 no goals
  -/


theorem AnalyticOnNhd.contDiff (hf : AnalyticOnNhd 𝕜 f univ) : ContDiff 𝕜 n f :=
  hf.analyticOn.contDiff


theorem ContDiff.analyticOnNhd (h : ContDiff 𝕜 ω f) : AnalyticOnNhd 𝕜 f s := by
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
    h : ContDiff 𝕜 Top.top f
    ⊢ AnalyticOnNhd 𝕜 f s
  -/
  rw [← contDiffOn_univ] at h
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
    h : ContDiffOn 𝕜 Top.top f Set.univ
    ⊢ AnalyticOnNhd 𝕜 f s
  -/
  have := h.analyticOn
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
    h : ContDiffOn 𝕜 Top.top f Set.univ
    this : AnalyticOn 𝕜 f Set.univ
    ⊢ AnalyticOnNhd 𝕜 f s
  -/
  rw [analyticOn_univ] at this
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
    h : ContDiffOn 𝕜 Top.top f Set.univ
    this : AnalyticOnNhd 𝕜 f Set.univ
    ⊢ AnalyticOnNhd 𝕜 f s
  -/
  exact this.mono (subset_univ _)
  /-
    🎉 no goals
  -/


theorem contDiff_omega_iff_analyticOnNhd :
    ContDiff 𝕜 ω f ↔ AnalyticOnNhd 𝕜 f univ :=
  ⟨fun h ↦ h.analyticOnNhd, fun h ↦ h.contDiff⟩


/-- When a function is `C^n`, it admits `ftaylorSeries 𝕜 f` as a Taylor series up
to order `n` in `s`. -/
theorem ContDiff.ftaylorSeries (hf : ContDiff 𝕜 n f) :
    HasFTaylorSeriesUpTo n f (ftaylorSeries 𝕜 f) := by
  simp only [← contDiffOn_univ, ← hasFTaylorSeriesUpToOn_univ_iff, ← ftaylorSeriesWithin_univ]
    at hf ⊢
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
    hf : ContDiffOn 𝕜 n f Set.univ
    ⊢ HasFTaylorSeriesUpToOn n f (ftaylorSeriesWithin 𝕜 f Set.univ) Set.univ
  -/
  exact ContDiffOn.ftaylorSeriesWithin hf uniqueDiffOn_univ
  /-
    🎉 no goals
  -/


/-- For `n : ℕ∞`, a function is `C^n` iff it admits `ftaylorSeries 𝕜 f`
as a Taylor series up to order `n`. -/
theorem contDiff_iff_ftaylorSeries {n : ℕ∞} :
    ContDiff 𝕜 n f ↔ HasFTaylorSeriesUpTo n f (ftaylorSeries 𝕜 f) := by
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
    n : ENat
    ⊢ Iff (ContDiff 𝕜 (↑n) f) (HasFTaylorSeriesUpTo (↑n) f (ftaylorSeries 𝕜 f))
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
      n : ENat
      ⊢ ContDiff 𝕜 (↑n) f → HasFTaylorSeriesUpTo (↑n) f (ftaylorSeries 𝕜 f)
    -/
  · rw [← contDiffOn_univ, ← hasFTaylorSeriesUpToOn_univ_iff, ← ftaylorSeriesWithin_univ]
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
      n : ENat
      ⊢ ContDiffOn 𝕜 (↑n) f Set.univ → HasFTaylorSeriesUpToOn (↑n) f (ftaylorSeriesW …
    -/
    exact fun h ↦ ContDiffOn.ftaylorSeriesWithin h uniqueDiffOn_univ
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
      n : ENat
      ⊢ HasFTaylorSeriesUpTo (↑n) f (ftaylorSeries 𝕜 f) → ContDiff 𝕜 (↑n) f
    -/
  · exact fun h ↦ ⟨ftaylorSeries 𝕜 f, h⟩
    /-
      🎉 no goals
    -/


theorem contDiff_iff_continuous_differentiable {n : ℕ∞} :
    ContDiff 𝕜 n f ↔
      (∀ m : ℕ, m ≤ n → Continuous fun x => iteratedFDeriv 𝕜 m f x) ∧
        ∀ m : ℕ, m < n → Differentiable 𝕜 fun x => iteratedFDeriv 𝕜 m f x := by
  simp [contDiffOn_univ.symm, continuous_iff_continuousOn_univ, differentiableOn_univ.symm,
    iteratedFDerivWithin_univ, contDiffOn_iff_continuousOn_differentiableOn uniqueDiffOn_univ]


theorem contDiff_nat_iff_continuous_differentiable {n : ℕ} :
    ContDiff 𝕜 n f ↔
      (∀ m : ℕ, m ≤ n → Continuous fun x => iteratedFDeriv 𝕜 m f x) ∧
        ∀ m : ℕ, m < n → Differentiable 𝕜 fun x => iteratedFDeriv 𝕜 m f x := by
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
    ⊢ Iff (ContDiff 𝕜 (↑n) f) (And (∀ (m : Nat), LE.le m n → Continuous fun x => i …
  -/
  rw [show n = ((n : ℕ∞) : WithTop ℕ∞) from rfl, contDiff_iff_continuous_differentiable]
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
    ⊢ Iff (And (∀ (m : Nat), LE.le ↑m ↑n → Continuous fun x => iteratedFDeriv 𝕜 m  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- If `f` is `C^n` then its `m`-times iterated derivative is continuous for `m ≤ n`. -/
theorem ContDiff.continuous_iteratedFDeriv {m : ℕ} (hm : m ≤ n) (hf : ContDiff 𝕜 n f) :
    Continuous fun x => iteratedFDeriv 𝕜 m f x :=
  (contDiff_iff_continuous_differentiable.mp (hf.of_le hm)).1 m le_rfl


/-- If `f` is `C^n` then its `m`-times iterated derivative is differentiable for `m < n`. -/
theorem ContDiff.differentiable_iteratedFDeriv {m : ℕ} (hm : m < n) (hf : ContDiff 𝕜 n f) :
    Differentiable 𝕜 fun x => iteratedFDeriv 𝕜 m f x :=
  (contDiff_iff_continuous_differentiable.mp
    (hf.of_le (ENat.add_one_natCast_le_withTop_of_lt hm))).2 m (mod_cast lt_add_one m)


theorem contDiff_of_differentiable_iteratedFDeriv {n : ℕ∞}
    (h : ∀ m : ℕ, m ≤ n → Differentiable 𝕜 (iteratedFDeriv 𝕜 m f)) : ContDiff 𝕜 n f :=
  contDiff_iff_continuous_differentiable.2
    ⟨fun m hm => (h m hm).continuous, fun m hm => h m (le_of_lt hm)⟩


/-- A function is `C^(n + 1)` if and only if it is differentiable,
and its derivative (formulated in terms of `fderiv`) is `C^n`. -/
theorem contDiff_succ_iff_fderiv :
    ContDiff 𝕜 (n + 1) f ↔ Differentiable 𝕜 f ∧ (n = ω → AnalyticOnNhd 𝕜 f univ) ∧
      ContDiff 𝕜 n (fderiv 𝕜 f) := by
  simp only [← contDiffOn_univ, ← differentiableOn_univ, ← fderivWithin_univ,
    contDiffOn_succ_iff_fderivWithin uniqueDiffOn_univ, analyticOn_univ]


theorem contDiff_one_iff_fderiv :
    ContDiff 𝕜 1 f ↔ Differentiable 𝕜 f ∧ Continuous (fderiv 𝕜 f) := by
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
    ⊢ Iff (ContDiff 𝕜 1 f) (And (Differentiable 𝕜 f) (Continuous (fderiv 𝕜 f)))
  -/
  rw [show (1 : WithTop ℕ∞) = 0 + 1 from rfl, contDiff_succ_iff_fderiv]
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
    ⊢ Iff (And (Differentiable 𝕜 f) (And (Eq 0 Top.top → AnalyticOnNhd 𝕜 f Set.uni …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem contDiff_infty_iff_fderiv :
    ContDiff 𝕜 ∞ f ↔ Differentiable 𝕜 f ∧ ContDiff 𝕜 ∞ (fderiv 𝕜 f) := by
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
    ⊢ Iff (ContDiff 𝕜 (↑Top.top) f) (And (Differentiable 𝕜 f) (ContDiff 𝕜 (↑Top.to …
  -/
  rw [show ∞ = ∞ + 1 from rfl, contDiff_succ_iff_fderiv]
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
    ⊢ Iff (And (Differentiable 𝕜 f) (And (Eq (↑Top.top) Top.top → AnalyticOnNhd 𝕜  …
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-11-27")] alias contDiff_top_iff_fderiv := contDiff_infty_iff_fderiv


theorem ContDiff.continuous_fderiv (h : ContDiff 𝕜 n f) (hn : 1 ≤ n) :
    Continuous (fderiv 𝕜 f) :=
  (contDiff_one_iff_fderiv.1 (h.of_le hn)).2


/-- If a function is at least `C^1`, its bundled derivative (mapping `(x, v)` to `Df(x) v`) is
continuous. -/
theorem ContDiff.continuous_fderiv_apply (h : ContDiff 𝕜 n f) (hn : 1 ≤ n) :
    Continuous fun p : E × E => (fderiv 𝕜 f p.1 : E → F) p.2 :=
  have A : Continuous fun q : (E →L[𝕜] F) × E => q.1 q.2 := isBoundedBilinearMap_apply.continuous
  have B : Continuous fun p : E × E => (fderiv 𝕜 f p.1, p.2) :=
    ((h.continuous_fderiv hn).comp continuous_fst).prod_mk continuous_snd
  A.comp B

