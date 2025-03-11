/-- **Tannery's theorem**: topological sums commute with termwise limits, when the norms of the
summands are eventually uniformly bounded by a summable function.

(This is the special case of the Lebesgue dominated convergence theorem for the counting measure
on a discrete set. However, we prove it under somewhat weaker assumptions than the general
measure-theoretic result, e.g. `G` is not assumed to be an `ℝ`-vector space or second countable,
and the limit is along an arbitrary filter rather than `atTop ℕ`.)

See also:
* `MeasureTheory.tendsto_integral_of_dominated_convergence` (for general integrals, but
  with more assumptions on `G`)
* `continuous_tsum` (continuity of infinite sums in a parameter)
-/
lemma tendsto_tsum_of_dominated_convergence {α β G : Type*} {𝓕 : Filter α}
    [NormedAddCommGroup G] [CompleteSpace G]
    {f : α → β → G} {g : β → G} {bound : β → ℝ} (h_sum : Summable bound)
    (hab : ∀ k : β, Tendsto (f · k) 𝓕 (𝓝 (g k)))
    (h_bound : ∀ᶠ n in 𝓕, ∀ k, ‖f n k‖ ≤ bound k) :
    Tendsto (∑' k, f · k) 𝓕 (𝓝 (∑' k, g k)) := by
  -- WLOG β is nonempty
  /-
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
    ⊢ Filter.Tendsto (fun x => tsum fun k => f x k) 𝓕 (nhds (tsum fun k => g k))
  -/
  rcases isEmpty_or_nonempty β
    /-
      case inl
      α : Type u_1
      β : Type u_2
      G : Type u_3
      𝓕 : Filter α
      inst✝¹ : NormedAddCommGroup G
      inst✝ : CompleteSpace G
      f : α → β → G
      g : β → G
      bound : β → Real
      h_sum : Summable bound
      hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
      h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
      h✝ : IsEmpty β
      ⊢ Filter.Tendsto (fun x => tsum fun k => f x k) 𝓕 (nhds (tsum fun k => g k))
    -/
  · simpa only [tsum_empty] using tendsto_const_nhds
    /-
      🎉 no goals
    -/
  -- WLOG 𝓕 ≠ ⊥
  /-
    case inr
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
    h✝ : Nonempty β
    ⊢ Filter.Tendsto (fun x => tsum fun k => f x k) 𝓕 (nhds (tsum fun k => g k))
  -/
  rcases 𝓕.eq_or_neBot with rfl | _
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      G : Type u_3
      inst✝¹ : NormedAddCommGroup G
      inst✝ : CompleteSpace G
      f : α → β → G
      g : β → G
      bound : β → Real
      h_sum : Summable bound
      h✝ : Nonempty β
      hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) Bot.bot (nhds (g k))
      h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
      ⊢ Filter.Tendsto (fun x => tsum fun k => f x k) Bot.bot (nhds (tsum fun k => g …
    -/
  · simp only [tendsto_bot]
    /-
      🎉 no goals
    -/
  -- Auxiliary lemmas
  have h_g_le (k : β) : ‖g k‖ ≤ bound k :=
    le_of_tendsto (tendsto_norm.comp (hab k)) <| h_bound.mono (fun n h => h k)
  have h_sumg : Summable (‖g ·‖) :=
    h_sum.of_norm_bounded _ (fun k ↦ (norm_norm (g k)).symm ▸ h_g_le k)
  have h_suma : ∀ᶠ n in 𝓕, Summable (‖f n ·‖) := by
    filter_upwards [h_bound] with n h
    exact h_sum.of_norm_bounded _ <| by simpa only [norm_norm] using h
  -- Now main proof, by an `ε / 3` argument
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
    h✝¹ : Nonempty β
    h✝ : 𝓕.NeBot
    h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
    h_sumg : Summable fun x => Norm.norm (g x)
    h_suma : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
    ⊢ Filter.Tendsto (fun x => tsum fun k => f x k) 𝓕 (nhds (tsum fun k => g k))
  -/
  rw [Metric.tendsto_nhds]
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
    h✝¹ : Nonempty β
    h✝ : 𝓕.NeBot
    h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
    h_sumg : Summable fun x => Norm.norm (g x)
    h_suma : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
    ⊢ ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (tsum …
  -/
  intro ε hε
  /-
    case inr.inr
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
    h✝¹ : Nonempty β
    h✝ : 𝓕.NeBot
    h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
    h_sumg : Summable fun x => Norm.norm (g x)
    h_suma : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
    ε : Real
    hε : GT.gt ε 0
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (tsum fun k => f x k) (tsum fun …
  -/
  let ⟨S, hS⟩ := h_sum
  obtain ⟨T, hT⟩ : ∃ (T : Finset β), dist (∑ b ∈ T, bound b) S < ε / 3 := by
    rw [HasSum, Metric.tendsto_nhds] at hS
    classical exact Eventually.exists <| hS _ (by positivity)
  have h1 : ∑' (k : (Tᶜ : Set β)), bound k < ε / 3 := by
    calc _ ≤ ‖∑' (k : (Tᶜ : Set β)), bound k‖ := Real.le_norm_self _
         _ = ‖S - ∑ b ∈ T, bound b‖          := congrArg _ ?_
         _ < ε / 3                            := by rwa [dist_eq_norm, norm_sub_rev] at hT
    simpa only [sum_add_tsum_compl h_sum, eq_sub_iff_add_eq'] using hS.tsum_eq
  /-
    case inr.inr.intro
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
    h✝¹ : Nonempty β
    h✝ : 𝓕.NeBot
    h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
    h_sumg : Summable fun x => Norm.norm (g x)
    h_suma : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
    ε : Real
    hε : GT.gt ε 0
    S : Real
    hS : HasSum bound S
    T : Finset β
    hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
    h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (tsum fun k => f x k) (tsum fun …
  -/
  have h2 : Tendsto (∑ k ∈ T, f · k) 𝓕 (𝓝 (T.sum g)) := tendsto_finset_sum _ (fun i _ ↦ hab i)
  /-
    case inr.inr.intro
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
    h✝¹ : Nonempty β
    h✝ : 𝓕.NeBot
    h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
    h_sumg : Summable fun x => Norm.norm (g x)
    h_suma : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
    ε : Real
    hε : GT.gt ε 0
    S : Real
    hS : HasSum bound S
    T : Finset β
    hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
    h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
    h2 : Filter.Tendsto (fun x => T.sum fun k => f x k) 𝓕 (nhds (T.sum g))
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (tsum fun k => f x k) (tsum fun …
  -/
  rw [Metric.tendsto_nhds] at h2
  /-
    case inr.inr.intro
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (bo …
    h✝¹ : Nonempty β
    h✝ : 𝓕.NeBot
    h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
    h_sumg : Summable fun x => Norm.norm (g x)
    h_suma : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
    ε : Real
    hε : GT.gt ε 0
    S : Real
    hS : HasSum bound S
    T : Finset β
    hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
    h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
    h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
    ⊢ Filter.Eventually (fun x => LT.lt (Dist.dist (tsum fun k => f x k) (tsum fun …
  -/
  filter_upwards [h2 (ε / 3) (by positivity), h_suma, h_bound] with n hn h_suma h_bound
  rw [dist_eq_norm, ← tsum_sub h_suma.of_norm h_sumg.of_norm,
    ← sum_add_tsum_compl (s := T) (h_suma.of_norm.sub h_sumg.of_norm),
    (by ring : ε = ε / 3 + (ε / 3 + ε / 3))]
  /-
    case h
    α : Type u_1
    β : Type u_2
    G : Type u_3
    𝓕 : Filter α
    inst✝¹ : NormedAddCommGroup G
    inst✝ : CompleteSpace G
    f : α → β → G
    g : β → G
    bound : β → Real
    h_sum : Summable bound
    hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
    h_bound✝ : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (b …
    h✝¹ : Nonempty β
    h✝ : 𝓕.NeBot
    h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
    h_sumg : Summable fun x => Norm.norm (g x)
    h_suma✝ : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
    ε : Real
    hε : GT.gt ε 0
    S : Real
    hS : HasSum bound S
    T : Finset β
    hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
    h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
    h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
    n : α
    hn : LT.lt (Dist.dist (T.sum fun k => f n k) (T.sum g)) (HDiv.hDiv ε 3)
    h_suma : Summable fun x => Norm.norm (f n x)
    h_bound : ∀ (k : β), LE.le (Norm.norm (f n k)) (bound k)
    ⊢ LT.lt (Norm.norm (HAdd.hAdd (T.sum fun x => HSub.hSub (f n x) (g x)) (tsum f …
  -/
  refine (norm_add_le _ _).trans_lt (add_lt_add ?_ ?_)
    /-
      case h.refine_1
      α : Type u_1
      β : Type u_2
      G : Type u_3
      𝓕 : Filter α
      inst✝¹ : NormedAddCommGroup G
      inst✝ : CompleteSpace G
      f : α → β → G
      g : β → G
      bound : β → Real
      h_sum : Summable bound
      hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
      h_bound✝ : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (b …
      h✝¹ : Nonempty β
      h✝ : 𝓕.NeBot
      h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
      h_sumg : Summable fun x => Norm.norm (g x)
      h_suma✝ : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
      ε : Real
      hε : GT.gt ε 0
      S : Real
      hS : HasSum bound S
      T : Finset β
      hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
      h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
      h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
      n : α
      hn : LT.lt (Dist.dist (T.sum fun k => f n k) (T.sum g)) (HDiv.hDiv ε 3)
      h_suma : Summable fun x => Norm.norm (f n x)
      h_bound : ∀ (k : β), LE.le (Norm.norm (f n k)) (bound k)
      ⊢ LT.lt (Norm.norm (T.sum fun x => HSub.hSub (f n x) (g x))) (HDiv.hDiv ε 3)
    -/
  · simpa only [dist_eq_norm, Finset.sum_sub_distrib] using hn
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      α : Type u_1
      β : Type u_2
      G : Type u_3
      𝓕 : Filter α
      inst✝¹ : NormedAddCommGroup G
      inst✝ : CompleteSpace G
      f : α → β → G
      g : β → G
      bound : β → Real
      h_sum : Summable bound
      hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
      h_bound✝ : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (b …
      h✝¹ : Nonempty β
      h✝ : 𝓕.NeBot
      h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
      h_sumg : Summable fun x => Norm.norm (g x)
      h_suma✝ : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
      ε : Real
      hε : GT.gt ε 0
      S : Real
      hS : HasSum bound S
      T : Finset β
      hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
      h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
      h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
      n : α
      hn : LT.lt (Dist.dist (T.sum fun k => f n k) (T.sum g)) (HDiv.hDiv ε 3)
      h_suma : Summable fun x => Norm.norm (f n x)
      h_bound : ∀ (k : β), LE.le (Norm.norm (f n k)) (bound k)
      ⊢ LT.lt (Norm.norm (tsum fun x => HSub.hSub (f n ↑x) (g ↑x))) (HAdd.hAdd (HDiv …
    -/
  · rw [tsum_sub (h_suma.subtype _).of_norm (h_sumg.subtype _).of_norm]
    /-
      case h.refine_2
      α : Type u_1
      β : Type u_2
      G : Type u_3
      𝓕 : Filter α
      inst✝¹ : NormedAddCommGroup G
      inst✝ : CompleteSpace G
      f : α → β → G
      g : β → G
      bound : β → Real
      h_sum : Summable bound
      hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
      h_bound✝ : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (b …
      h✝¹ : Nonempty β
      h✝ : 𝓕.NeBot
      h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
      h_sumg : Summable fun x => Norm.norm (g x)
      h_suma✝ : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
      ε : Real
      hε : GT.gt ε 0
      S : Real
      hS : HasSum bound S
      T : Finset β
      hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
      h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
      h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
      n : α
      hn : LT.lt (Dist.dist (T.sum fun k => f n k) (T.sum g)) (HDiv.hDiv ε 3)
      h_suma : Summable fun x => Norm.norm (f n x)
      h_bound : ∀ (k : β), LE.le (Norm.norm (f n k)) (bound k)
      ⊢ LT.lt (Norm.norm (HSub.hSub (tsum fun b => f n ↑b) (tsum fun b => g ↑b))) (H …
    -/
    refine (norm_sub_le _ _).trans_lt (add_lt_add ?_ ?_)
      /-
        case h.refine_2.refine_1
        α : Type u_1
        β : Type u_2
        G : Type u_3
        𝓕 : Filter α
        inst✝¹ : NormedAddCommGroup G
        inst✝ : CompleteSpace G
        f : α → β → G
        g : β → G
        bound : β → Real
        h_sum : Summable bound
        hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
        h_bound✝ : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (b …
        h✝¹ : Nonempty β
        h✝ : 𝓕.NeBot
        h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
        h_sumg : Summable fun x => Norm.norm (g x)
        h_suma✝ : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
        ε : Real
        hε : GT.gt ε 0
        S : Real
        hS : HasSum bound S
        T : Finset β
        hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
        h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
        h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
        n : α
        hn : LT.lt (Dist.dist (T.sum fun k => f n k) (T.sum g)) (HDiv.hDiv ε 3)
        h_suma : Summable fun x => Norm.norm (f n x)
        h_bound : ∀ (k : β), LE.le (Norm.norm (f n k)) (bound k)
        ⊢ LT.lt (Norm.norm (tsum fun b => f n ↑b)) (HDiv.hDiv ε 3)
      -/
    · refine ((norm_tsum_le_tsum_norm (h_suma.subtype _)).trans ?_).trans_lt h1
      /-
        case h.refine_2.refine_1
        α : Type u_1
        β : Type u_2
        G : Type u_3
        𝓕 : Filter α
        inst✝¹ : NormedAddCommGroup G
        inst✝ : CompleteSpace G
        f : α → β → G
        g : β → G
        bound : β → Real
        h_sum : Summable bound
        hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
        h_bound✝ : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (b …
        h✝¹ : Nonempty β
        h✝ : 𝓕.NeBot
        h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
        h_sumg : Summable fun x => Norm.norm (g x)
        h_suma✝ : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
        ε : Real
        hε : GT.gt ε 0
        S : Real
        hS : HasSum bound S
        T : Finset β
        hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
        h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
        h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
        n : α
        hn : LT.lt (Dist.dist (T.sum fun k => f n k) (T.sum g)) (HDiv.hDiv ε 3)
        h_suma : Summable fun x => Norm.norm (f n x)
        h_bound : ∀ (k : β), LE.le (Norm.norm (f n k)) (bound k)
        ⊢ LE.le (tsum fun i => Norm.norm (f n ↑i)) (tsum fun k => bound ↑k)
      -/
      exact tsum_le_tsum (h_bound ·) (h_suma.subtype _) (h_sum.subtype _)
      /-
        🎉 no goals
      -/
      /-
        case h.refine_2.refine_2
        α : Type u_1
        β : Type u_2
        G : Type u_3
        𝓕 : Filter α
        inst✝¹ : NormedAddCommGroup G
        inst✝ : CompleteSpace G
        f : α → β → G
        g : β → G
        bound : β → Real
        h_sum : Summable bound
        hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
        h_bound✝ : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (b …
        h✝¹ : Nonempty β
        h✝ : 𝓕.NeBot
        h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
        h_sumg : Summable fun x => Norm.norm (g x)
        h_suma✝ : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
        ε : Real
        hε : GT.gt ε 0
        S : Real
        hS : HasSum bound S
        T : Finset β
        hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
        h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
        h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
        n : α
        hn : LT.lt (Dist.dist (T.sum fun k => f n k) (T.sum g)) (HDiv.hDiv ε 3)
        h_suma : Summable fun x => Norm.norm (f n x)
        h_bound : ∀ (k : β), LE.le (Norm.norm (f n k)) (bound k)
        ⊢ LT.lt (Norm.norm (tsum fun b => g ↑b)) (HDiv.hDiv ε 3)
      -/
    · refine ((norm_tsum_le_tsum_norm <| h_sumg.subtype _).trans ?_).trans_lt h1
      /-
        case h.refine_2.refine_2
        α : Type u_1
        β : Type u_2
        G : Type u_3
        𝓕 : Filter α
        inst✝¹ : NormedAddCommGroup G
        inst✝ : CompleteSpace G
        f : α → β → G
        g : β → G
        bound : β → Real
        h_sum : Summable bound
        hab : ∀ (k : β), Filter.Tendsto (fun x => f x k) 𝓕 (nhds (g k))
        h_bound✝ : Filter.Eventually (fun n => ∀ (k : β), LE.le (Norm.norm (f n k)) (b …
        h✝¹ : Nonempty β
        h✝ : 𝓕.NeBot
        h_g_le : ∀ (k : β), LE.le (Norm.norm (g k)) (bound k)
        h_sumg : Summable fun x => Norm.norm (g x)
        h_suma✝ : Filter.Eventually (fun n => Summable fun x => Norm.norm (f n x)) 𝓕
        ε : Real
        hε : GT.gt ε 0
        S : Real
        hS : HasSum bound S
        T : Finset β
        hT : LT.lt (Dist.dist (T.sum fun b => bound b) S) (HDiv.hDiv ε 3)
        h1 : LT.lt (tsum fun k => bound ↑k) (HDiv.hDiv ε 3)
        h2 : ∀ (ε : Real), GT.gt ε 0 → Filter.Eventually (fun x => LT.lt (Dist.dist (T …
        n : α
        hn : LT.lt (Dist.dist (T.sum fun k => f n k) (T.sum g)) (HDiv.hDiv ε 3)
        h_suma : Summable fun x => Norm.norm (f n x)
        h_bound : ∀ (k : β), LE.le (Norm.norm (f n k)) (bound k)
        ⊢ LE.le (tsum fun i => Norm.norm (g ↑i)) (tsum fun k => bound ↑k)
      -/
      exact tsum_le_tsum (h_g_le ·) (h_sumg.subtype _) (h_sum.subtype _)
      /-
        🎉 no goals
      -/

