/-- The gauge rescale map `gaugeRescale s t` sends each point `x` to the point `y` on the same ray
that has the same gauge w.r.t. `t` as `x` has w.r.t. `s`.

The characteristic property is satisfied if `gauge t x ≠ 0`, see `gauge_gaugeRescale'`.
In particular, it is satisfied for all `x`,
provided that `t` is absorbent and von Neumann bounded. -/
def gaugeRescale (s t : Set E) (x : E) : E := (gauge s x / gauge t x) • x


theorem gaugeRescale_def (s t : Set E) (x : E) :
    gaugeRescale s t x = (gauge s x / gauge t x) • x :=
  rfl


@[simp] theorem gaugeRescale_zero (s t : Set E) : gaugeRescale s t 0 = 0 := smul_zero _


theorem gaugeRescale_smul (s t : Set E) {c : ℝ} (hc : 0 ≤ c) (x : E) :
    gaugeRescale s t (c • x) = c • gaugeRescale s t x := by
  /-
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s t : Set E
    c : Real
    hc : LE.le 0 c
    x : E
    ⊢ Eq (gaugeRescale s t (HSMul.hSMul c x)) (HSMul.hSMul c (gaugeRescale s t x))
  -/
  simp only [gaugeRescale, gauge_smul_of_nonneg hc, smul_smul, smul_eq_mul]
  /-
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s t : Set E
    c : Real
    hc : LE.le 0 c
    x : E
    ⊢ Eq (HSMul.hSMul (HMul.hMul (HDiv.hDiv (HMul.hMul c (gauge s x)) (HMul.hMul c …
  -/
  rw [mul_div_mul_comm, mul_right_comm, div_self_mul_self]
  /-
    🎉 no goals
  -/


theorem gauge_gaugeRescale' (s : Set E) {t : Set E} {x : E} (hx : gauge t x ≠ 0) :
    gauge t (gaugeRescale s t x) = gauge s x := by
  rw [gaugeRescale, gauge_smul_of_nonneg (div_nonneg (gauge_nonneg _) (gauge_nonneg _)),
    smul_eq_mul, div_mul_cancel₀ _ hx]


theorem gauge_gaugeRescale_le (s t : Set E) (x : E) :
    gauge t (gaugeRescale s t x) ≤ gauge s x := by
  /-
    E : Type u_1
    inst✝¹ : AddCommGroup E
    inst✝ : Module Real E
    s t : Set E
    x : E
    ⊢ LE.le (gauge t (gaugeRescale s t x)) (gauge s x)
  -/
  by_cases hx : gauge t x = 0
    /-
      case pos
      E : Type u_1
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s t : Set E
      x : E
      hx : Eq (gauge t x) 0
      ⊢ LE.le (gauge t (gaugeRescale s t x)) (gauge s x)
    -/
  · simp [gaugeRescale, hx, gauge_nonneg]
    /-
      🎉 no goals
    -/
    /-
      case neg
      E : Type u_1
      inst✝¹ : AddCommGroup E
      inst✝ : Module Real E
      s t : Set E
      x : E
      hx : Not (Eq (gauge t x) 0)
      ⊢ LE.le (gauge t (gaugeRescale s t x)) (gauge s x)
    -/
  · exact (gauge_gaugeRescale' s hx).le
    /-
      🎉 no goals
    -/


theorem gaugeRescale_self_apply {s : Set E} (hsa : Absorbent ℝ s) (hsb : IsVonNBounded ℝ s)
    (x : E) : gaugeRescale s s x = x := by
  /-
    E : Type u_1
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    s : Set E
    hsa : Absorbent Real s
    hsb : Bornology.IsVonNBounded Real s
    x : E
    ⊢ Eq (gaugeRescale s s x) x
  -/
  rcases eq_or_ne x 0 with rfl | hx; · simp
                                       /-
                                         🎉 no goals
                                       -/
  /-
    case inr
    E : Type u_1
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    s : Set E
    hsa : Absorbent Real s
    hsb : Bornology.IsVonNBounded Real s
    x : E
    hx : Ne x 0
    ⊢ Eq (gaugeRescale s s x) x
  -/
  rw [gaugeRescale, div_self, one_smul]
  /-
    case inr
    E : Type u_1
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    s : Set E
    hsa : Absorbent Real s
    hsb : Bornology.IsVonNBounded Real s
    x : E
    hx : Ne x 0
    ⊢ Ne (gauge s x) 0
  -/
  exact ((gauge_pos hsa hsb).2 hx).ne'
  /-
    🎉 no goals
  -/


theorem gaugeRescale_self {s : Set E} (hsa : Absorbent ℝ s) (hsb : IsVonNBounded ℝ s) :
    gaugeRescale s s = id :=
  funext <| gaugeRescale_self_apply hsa hsb


theorem gauge_gaugeRescale (s : Set E) {t : Set E} (hta : Absorbent ℝ t) (htb : IsVonNBounded ℝ t)
    (x : E) : gauge t (gaugeRescale s t x) = gauge s x := by
  /-
    E : Type u_1
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    s t : Set E
    hta : Absorbent Real t
    htb : Bornology.IsVonNBounded Real t
    x : E
    ⊢ Eq (gauge t (gaugeRescale s t x)) (gauge s x)
  -/
  rcases eq_or_ne x 0 with rfl | hx
    /-
      case inl
      E : Type u_1
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      inst✝¹ : TopologicalSpace E
      inst✝ : T1Space E
      s t : Set E
      hta : Absorbent Real t
      htb : Bornology.IsVonNBounded Real t
      ⊢ Eq (gauge t (gaugeRescale s t 0)) (gauge s 0)
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      E : Type u_1
      inst✝³ : AddCommGroup E
      inst✝² : Module Real E
      inst✝¹ : TopologicalSpace E
      inst✝ : T1Space E
      s t : Set E
      hta : Absorbent Real t
      htb : Bornology.IsVonNBounded Real t
      x : E
      hx : Ne x 0
      ⊢ Eq (gauge t (gaugeRescale s t x)) (gauge s x)
    -/
  · exact gauge_gaugeRescale' s ((gauge_pos hta htb).2 hx).ne'
    /-
      🎉 no goals
    -/


theorem gaugeRescale_gaugeRescale {s t u : Set E} (hta : Absorbent ℝ t) (htb : IsVonNBounded ℝ t)
    (x : E) : gaugeRescale t u (gaugeRescale s t x) = gaugeRescale s u x := by
  /-
    E : Type u_1
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    s t u : Set E
    hta : Absorbent Real t
    htb : Bornology.IsVonNBounded Real t
    x : E
    ⊢ Eq (gaugeRescale t u (gaugeRescale s t x)) (gaugeRescale s u x)
  -/
  rcases eq_or_ne x 0 with rfl | hx; · simp
                                       /-
                                         🎉 no goals
                                       -/
  rw [gaugeRescale_def s t x, gaugeRescale_smul, gaugeRescale, gaugeRescale, smul_smul,
    div_mul_div_cancel₀]
  /-
    case inr
    E : Type u_1
    inst✝³ : AddCommGroup E
    inst✝² : Module Real E
    inst✝¹ : TopologicalSpace E
    inst✝ : T1Space E
    s t u : Set E
    hta : Absorbent Real t
    htb : Bornology.IsVonNBounded Real t
    x : E
    hx : Ne x 0
    ⊢ Ne (gauge t x) 0
  -/
  exacts [((gauge_pos hta htb).2 hx).ne', div_nonneg (gauge_nonneg _) (gauge_nonneg _)]
  /-
    🎉 no goals
  -/


/-- `gaugeRescale` bundled as an `Equiv`. -/
def gaugeRescaleEquiv (s t : Set E) (hsa : Absorbent ℝ s) (hsb : IsVonNBounded ℝ s)
    (hta : Absorbent ℝ t) (htb : IsVonNBounded ℝ t) : E ≃ E where
  toFun := gaugeRescale s t
  invFun := gaugeRescale t s
                   /-
                     E : Type u_1
                     inst✝³ : AddCommGroup E
                     inst✝² : Module Real E
                     inst✝¹ : TopologicalSpace E
                     inst✝ : T1Space E
                     s t : Set E
                     hsa : Absorbent Real s
                     hsb : Bornology.IsVonNBounded Real s
                     hta : Absorbent Real t
                     htb : Bornology.IsVonNBounded Real t
                     x : E
                     ⊢ Eq (gaugeRescale t s (gaugeRescale s t x)) x
                   -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  left_inv x := by rw [gaugeRescale_gaugeRescale, gaugeRescale_self_apply] <;> assumption
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
                    /-
                      E : Type u_1
                      inst✝³ : AddCommGroup E
                      inst✝² : Module Real E
                      inst✝¹ : TopologicalSpace E
                      inst✝ : T1Space E
                      s t : Set E
                      hsa : Absorbent Real s
                      hsb : Bornology.IsVonNBounded Real s
                      hta : Absorbent Real t
                      htb : Bornology.IsVonNBounded Real t
                      x : E
                      ⊢ Eq (gaugeRescale s t (gaugeRescale t s x)) x
                    -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/
  right_inv x := by rw [gaugeRescale_gaugeRescale, gaugeRescale_self_apply] <;> assumption
                                                                                /-
                                                                                  🎉 no goals
                                                                                -/


theorem mapsTo_gaugeRescale_interior (h₀ : t ∈ 𝓝 0) (hc : Convex ℝ t) :
    MapsTo (gaugeRescale s t) (interior s) (interior t) := fun x hx ↦ by
  /-
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s t : Set E
    h₀ : Membership.mem (nhds 0) t
    hc : Convex Real t
    x : E
    hx : Membership.mem (interior s) x
    ⊢ Membership.mem (interior t) (gaugeRescale s t x)
  -/
                                           /-
                                             🎉 no goals
                                           -/
  rw [← gauge_lt_one_iff_mem_interior] <;> try assumption
                                           /-
                                             🎉 no goals
                                           -/
  /-
    E : Type u_1
    inst✝⁴ : AddCommGroup E
    inst✝³ : Module Real E
    inst✝² : TopologicalSpace E
    inst✝¹ : TopologicalAddGroup E
    inst✝ : ContinuousSMul Real E
    s t : Set E
    h₀ : Membership.mem (nhds 0) t
    hc : Convex Real t
    x : E
    hx : Membership.mem (interior s) x
    ⊢ LT.lt (gauge t (gaugeRescale s t x)) 1
  -/
  exact (gauge_gaugeRescale_le _ _ _).trans_lt (interior_subset_gauge_lt_one _ hx)
  /-
    🎉 no goals
  -/


theorem mapsTo_gaugeRescale_closure {s t : Set E} (hsc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0)
    (htc : Convex ℝ t) (ht₀ : 0 ∈ t) (hta : Absorbent ℝ t) :
    MapsTo (gaugeRescale s t) (closure s) (closure t) := fun _x hx ↦
  mem_closure_of_gauge_le_one htc ht₀ hta <| (gauge_gaugeRescale_le _ _ _).trans <|
    (gauge_le_one_iff_mem_closure hsc hs₀).2 hx


theorem continuous_gaugeRescale {s t : Set E} (hs : Convex ℝ s) (hs₀ : s ∈ 𝓝 0)
    (ht : Convex ℝ t) (ht₀ : t ∈ 𝓝 0) (htb : IsVonNBounded ℝ t) :
    Continuous (gaugeRescale s t) := by
  /-
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T1Space E
    s t : Set E
    hs : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ht : Convex Real t
    ht₀ : Membership.mem (nhds 0) t
    htb : Bornology.IsVonNBounded Real t
    ⊢ Continuous (gaugeRescale s t)
  -/
  have hta : Absorbent ℝ t := absorbent_nhds_zero ht₀
  /-
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T1Space E
    s t : Set E
    hs : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ht : Convex Real t
    ht₀ : Membership.mem (nhds 0) t
    htb : Bornology.IsVonNBounded Real t
    hta : Absorbent Real t
    ⊢ Continuous (gaugeRescale s t)
  -/
  refine continuous_iff_continuousAt.2 fun x ↦ ?_
  /-
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T1Space E
    s t : Set E
    hs : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    ht : Convex Real t
    ht₀ : Membership.mem (nhds 0) t
    htb : Bornology.IsVonNBounded Real t
    hta : Absorbent Real t
    x : E
    ⊢ ContinuousAt (gaugeRescale s t) x
  -/
  rcases eq_or_ne x 0 with rfl | hx
    /-
      case inl
      E : Type u_1
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Real E
      inst✝³ : TopologicalSpace E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : T1Space E
      s t : Set E
      hs : Convex Real s
      hs₀ : Membership.mem (nhds 0) s
      ht : Convex Real t
      ht₀ : Membership.mem (nhds 0) t
      htb : Bornology.IsVonNBounded Real t
      hta : Absorbent Real t
      ⊢ ContinuousAt (gaugeRescale s t) 0
    -/
  · rw [ContinuousAt, gaugeRescale_zero]
    /-
      case inl
      E : Type u_1
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Real E
      inst✝³ : TopologicalSpace E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : T1Space E
      s t : Set E
      hs : Convex Real s
      hs₀ : Membership.mem (nhds 0) s
      ht : Convex Real t
      ht₀ : Membership.mem (nhds 0) t
      htb : Bornology.IsVonNBounded Real t
      hta : Absorbent Real t
      ⊢ Filter.Tendsto (gaugeRescale s t) (nhds 0) (nhds 0)
    -/
    nth_rewrite 2 [← comap_gauge_nhds_zero htb ht₀]
    /-
      case inl
      E : Type u_1
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Real E
      inst✝³ : TopologicalSpace E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : T1Space E
      s t : Set E
      hs : Convex Real s
      hs₀ : Membership.mem (nhds 0) s
      ht : Convex Real t
      ht₀ : Membership.mem (nhds 0) t
      htb : Bornology.IsVonNBounded Real t
      hta : Absorbent Real t
      ⊢ Filter.Tendsto (gaugeRescale s t) (nhds 0) (Filter.comap (gauge t) (nhds 0))
    -/
    simp only [tendsto_comap_iff, Function.comp_def, gauge_gaugeRescale _ hta htb]
    /-
      case inl
      E : Type u_1
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Real E
      inst✝³ : TopologicalSpace E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : T1Space E
      s t : Set E
      hs : Convex Real s
      hs₀ : Membership.mem (nhds 0) s
      ht : Convex Real t
      ht₀ : Membership.mem (nhds 0) t
      htb : Bornology.IsVonNBounded Real t
      hta : Absorbent Real t
      ⊢ Filter.Tendsto (fun x => gauge s x) (nhds 0) (nhds 0)
    -/
    exact tendsto_gauge_nhds_zero hs₀
    /-
      🎉 no goals
    -/
  · exact ((continuousAt_gauge hs hs₀).div (continuousAt_gauge ht ht₀)
      ((gauge_pos hta htb).2 hx).ne').smul continuousAt_id


/-- `gaugeRescale` bundled as a `Homeomorph`. -/
def gaugeRescaleHomeomorph (s t : Set E)
    (hsc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) (hsb : IsVonNBounded ℝ s)
    (htc : Convex ℝ t) (ht₀ : t ∈ 𝓝 0) (htb : IsVonNBounded ℝ t) : E ≃ₜ E where
  toEquiv := gaugeRescaleEquiv s t (absorbent_nhds_zero hs₀) hsb (absorbent_nhds_zero ht₀) htb
                         /-
                           E : Type u_1
                           inst✝⁵ : AddCommGroup E
                           inst✝⁴ : Module Real E
                           inst✝³ : TopologicalSpace E
                           inst✝² : TopologicalAddGroup E
                           inst✝¹ : ContinuousSMul Real E
                           s✝ t✝ : Set E
                           inst✝ : T1Space E
                           s t : Set E
                           hsc : Convex Real s
                           hs₀ : Membership.mem (nhds 0) s
                           hsb : Bornology.IsVonNBounded Real s
                           htc : Convex Real t
                           ht₀ : Membership.mem (nhds 0) t
                           htb : Bornology.IsVonNBounded Real t
                           ⊢ Continuous (gaugeRescaleEquiv s t ⋯ hsb ⋯ htb).toFun
                         -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
                                                           /-
                                                             🎉 no goals
                                                           -/
  continuous_toFun := by apply continuous_gaugeRescale <;> assumption
                                                           /-
                                                             🎉 no goals
                                                           -/
                          /-
                            E : Type u_1
                            inst✝⁵ : AddCommGroup E
                            inst✝⁴ : Module Real E
                            inst✝³ : TopologicalSpace E
                            inst✝² : TopologicalAddGroup E
                            inst✝¹ : ContinuousSMul Real E
                            s✝ t✝ : Set E
                            inst✝ : T1Space E
                            s t : Set E
                            hsc : Convex Real s
                            hs₀ : Membership.mem (nhds 0) s
                            hsb : Bornology.IsVonNBounded Real s
                            htc : Convex Real t
                            ht₀ : Membership.mem (nhds 0) t
                            htb : Bornology.IsVonNBounded Real t
                            ⊢ Continuous (gaugeRescaleEquiv s t ⋯ hsb ⋯ htb).invFun
                          -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
                                                            /-
                                                              🎉 no goals
                                                            -/
  continuous_invFun := by apply continuous_gaugeRescale <;> assumption
                                                            /-
                                                              🎉 no goals
                                                            -/


theorem image_gaugeRescaleHomeomorph_interior {s t : Set E}
    (hsc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) (hsb : IsVonNBounded ℝ s)
    (htc : Convex ℝ t) (ht₀ : t ∈ 𝓝 0) (htb : IsVonNBounded ℝ t) :
    gaugeRescaleHomeomorph s t hsc hs₀ hsb htc ht₀ htb '' interior s = interior t :=
  Subset.antisymm (mapsTo_gaugeRescale_interior ht₀ htc).image_subset <| by
    /-
      E : Type u_1
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Real E
      inst✝³ : TopologicalSpace E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : T1Space E
      s t : Set E
      hsc : Convex Real s
      hs₀ : Membership.mem (nhds 0) s
      hsb : Bornology.IsVonNBounded Real s
      htc : Convex Real t
      ht₀ : Membership.mem (nhds 0) t
      htb : Bornology.IsVonNBounded Real t
      ⊢ HasSubset.Subset (interior t) (Set.image (⇑(gaugeRescaleHomeomorph s t hsc h …
    -/
    rw [← Homeomorph.preimage_symm, ← image_subset_iff]
    /-
      E : Type u_1
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Real E
      inst✝³ : TopologicalSpace E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : T1Space E
      s t : Set E
      hsc : Convex Real s
      hs₀ : Membership.mem (nhds 0) s
      hsb : Bornology.IsVonNBounded Real s
      htc : Convex Real t
      ht₀ : Membership.mem (nhds 0) t
      htb : Bornology.IsVonNBounded Real t
      ⊢ HasSubset.Subset (Set.image (⇑(gaugeRescaleHomeomorph s t hsc hs₀ hsb htc ht …
    -/
    exact (mapsTo_gaugeRescale_interior hs₀ hsc).image_subset
    /-
      🎉 no goals
    -/


theorem image_gaugeRescaleHomeomorph_closure {s t : Set E}
    (hsc : Convex ℝ s) (hs₀ : s ∈ 𝓝 0) (hsb : IsVonNBounded ℝ s)
    (htc : Convex ℝ t) (ht₀ : t ∈ 𝓝 0) (htb : IsVonNBounded ℝ t) :
    gaugeRescaleHomeomorph s t hsc hs₀ hsb htc ht₀ htb '' closure s = closure t := by
  refine Subset.antisymm (mapsTo_gaugeRescale_closure hsc hs₀ htc
    (mem_of_mem_nhds ht₀) (absorbent_nhds_zero ht₀)).image_subset ?_
  /-
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T1Space E
    s t : Set E
    hsc : Convex Real s
    hs₀ : Membership.mem (nhds 0) s
    hsb : Bornology.IsVonNBounded Real s
    htc : Convex Real t
    ht₀ : Membership.mem (nhds 0) t
    htb : Bornology.IsVonNBounded Real t
    ⊢ HasSubset.Subset (closure t) (Set.image (⇑(gaugeRescaleHomeomorph s t hsc hs …
  -/
  rw [← Homeomorph.preimage_symm, ← image_subset_iff]
  exact (mapsTo_gaugeRescale_closure htc ht₀ hsc
    (mem_of_mem_nhds hs₀) (absorbent_nhds_zero hs₀)).image_subset


/-- Given two convex bounded sets in a topological vector space with nonempty interiors,
there exists a homeomorphism of the ambient space
that sends the interior, the closure, and the frontier of one set
to the interior, the closure, and the frontier of the other set.

In particular, if both `s` and `t` are open set or both `s` and `t` are closed sets,
then `e` maps `s` to `t`. -/
theorem exists_homeomorph_image_eq {s t : Set E}
    (hsc : Convex ℝ s) (hsne : (interior s).Nonempty) (hsb : IsVonNBounded ℝ s)
    (hst : Convex ℝ t) (htne : (interior t).Nonempty) (htb : IsVonNBounded ℝ t) :
    ∃ e : E ≃ₜ E, e '' interior s = interior t ∧ e '' closure s = closure t ∧
      e '' frontier s = frontier t := by
  /-
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T1Space E
    s t : Set E
    hsc : Convex Real s
    hsne : (interior s).Nonempty
    hsb : Bornology.IsVonNBounded Real s
    hst : Convex Real t
    htne : (interior t).Nonempty
    htb : Bornology.IsVonNBounded Real t
    ⊢ Exists fun e => And (Eq (Set.image (⇑e) (interior s)) (interior t)) (And (Eq …
  -/
  rsuffices ⟨e, h₁, h₂⟩ : ∃ e : E ≃ₜ E, e '' interior s = interior t ∧ e '' closure s = closure t
    /-
      case intro.intro
      E : Type u_1
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Real E
      inst✝³ : TopologicalSpace E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : T1Space E
      s t : Set E
      hsc : Convex Real s
      hsne : (interior s).Nonempty
      hsb : Bornology.IsVonNBounded Real s
      hst : Convex Real t
      htne : (interior t).Nonempty
      htb : Bornology.IsVonNBounded Real t
      e : Homeomorph E E
      h₁ : Eq (Set.image (⇑e) (interior s)) (interior t)
      h₂ : Eq (Set.image (⇑e) (closure s)) (closure t)
      ⊢ Exists fun e => And (Eq (Set.image (⇑e) (interior s)) (interior t)) (And (Eq …
    -/
  · refine ⟨e, h₁, h₂, ?_⟩
    /-
      case intro.intro
      E : Type u_1
      inst✝⁵ : AddCommGroup E
      inst✝⁴ : Module Real E
      inst✝³ : TopologicalSpace E
      inst✝² : TopologicalAddGroup E
      inst✝¹ : ContinuousSMul Real E
      inst✝ : T1Space E
      s t : Set E
      hsc : Convex Real s
      hsne : (interior s).Nonempty
      hsb : Bornology.IsVonNBounded Real s
      hst : Convex Real t
      htne : (interior t).Nonempty
      htb : Bornology.IsVonNBounded Real t
      e : Homeomorph E E
      h₁ : Eq (Set.image (⇑e) (interior s)) (interior t)
      h₂ : Eq (Set.image (⇑e) (closure s)) (closure t)
      ⊢ Eq (Set.image (⇑e) (frontier s)) (frontier t)
    -/
    simp_rw [← closure_diff_interior, image_diff e.injective, h₁, h₂]
    /-
      🎉 no goals
    -/
  /-
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T1Space E
    s t : Set E
    hsc : Convex Real s
    hsne : (interior s).Nonempty
    hsb : Bornology.IsVonNBounded Real s
    hst : Convex Real t
    htne : (interior t).Nonempty
    htb : Bornology.IsVonNBounded Real t
    ⊢ Exists fun e => And (Eq (Set.image (⇑e) (interior s)) (interior t)) (Eq (Set …
  -/
  rcases hsne with ⟨x, hx⟩
  /-
    case intro
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T1Space E
    s t : Set E
    hsc : Convex Real s
    hsb : Bornology.IsVonNBounded Real s
    hst : Convex Real t
    htne : (interior t).Nonempty
    htb : Bornology.IsVonNBounded Real t
    x : E
    hx : Membership.mem (interior s) x
    ⊢ Exists fun e => And (Eq (Set.image (⇑e) (interior s)) (interior t)) (Eq (Set …
  -/
  rcases htne with ⟨y, hy⟩
  set h : E ≃ₜ E := by
    apply gaugeRescaleHomeomorph (-x +ᵥ s) (-y +ᵥ t) <;>
      simp [← mem_interior_iff_mem_nhds, interior_vadd, mem_vadd_set_iff_neg_vadd_mem, *]
  /-
    case intro.intro
    E : Type u_1
    inst✝⁵ : AddCommGroup E
    inst✝⁴ : Module Real E
    inst✝³ : TopologicalSpace E
    inst✝² : TopologicalAddGroup E
    inst✝¹ : ContinuousSMul Real E
    inst✝ : T1Space E
    s t : Set E
    hsc : Convex Real s
    hsb : Bornology.IsVonNBounded Real s
    hst : Convex Real t
    htb : Bornology.IsVonNBounded Real t
    x : E
    hx : Membership.mem (interior s) x
    y : E
    hy : Membership.mem (interior t) y
    h : Homeomorph E E := gaugeRescaleHomeomorph (HVAdd.hVAdd (Neg.neg x) s) (HVAd …
    ⊢ Exists fun e => And (Eq (Set.image (⇑e) (interior s)) (interior t)) (Eq (Set …
  -/
  refine ⟨.trans (.addLeft (-x)) <| h.trans <| .addLeft y, ?_, ?_⟩
  · calc
      (fun a ↦ y + h (-x + a)) '' interior s = y +ᵥ h '' interior (-x +ᵥ s) := by
        simp_rw [interior_vadd, ← image_vadd, image_image, vadd_eq_add]
      _ = _ := by rw [image_gaugeRescaleHomeomorph_interior, interior_vadd, vadd_neg_vadd]
  · calc
      (fun a ↦ y + h (-x + a)) '' closure s = y +ᵥ h '' closure (-x +ᵥ s) := by
        simp_rw [closure_vadd, ← image_vadd, image_image, vadd_eq_add]
      _ = _ := by rw [image_gaugeRescaleHomeomorph_closure, closure_vadd, vadd_neg_vadd]


/-- If `s` is a convex bounded set with a nonempty interior in a real normed space,
then there is a homeomorphism of the ambient space to itself
that sends the interior of `s` to the unit open ball
and the closure of `s` to the unit closed ball. -/
theorem exists_homeomorph_image_interior_closure_frontier_eq_unitBall {s : Set E}
    (hc : Convex ℝ s) (hne : (interior s).Nonempty) (hb : IsBounded s) :
    ∃ h : E ≃ₜ E, h '' interior s = ball 0 1 ∧ h '' closure s = closedBall 0 1 ∧
      h '' frontier s = sphere 0 1 := by
  simpa [isOpen_ball.interior_eq, closure_ball, frontier_ball]
    using exists_homeomorph_image_eq hc hne (NormedSpace.isVonNBounded_of_isBounded _ hb)
    (convex_ball 0 1) (by simp [isOpen_ball.interior_eq]) (NormedSpace.isVonNBounded_ball _ _ _)

