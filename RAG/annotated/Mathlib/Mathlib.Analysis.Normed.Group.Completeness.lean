lemma Metric.exists_subseq_summable_dist_of_cauchySeq (u : ℕ → α) (hu : CauchySeq u) :
    ∃ f : ℕ → ℕ, StrictMono f ∧ Summable fun i => dist (u (f (i+1))) (u (f i)) := by
  obtain ⟨f, hf₁, hf₂⟩ := Metric.exists_subseq_bounded_of_cauchySeq u hu
    (fun n => (1 / (2 : ℝ))^n) (fun n => by positivity)
  /-
    case intro.intro
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : ∀ (n m : Nat), GE.ge m (f n) → LT.lt (Dist.dist (u m) (u (f n))) (HPow.h …
    ⊢ Exists fun f => And (StrictMono f) (Summable fun i => Dist.dist (u (f (HAdd. …
  -/
  refine ⟨f, hf₁, ?_⟩
  /-
    case intro.intro
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : ∀ (n m : Nat), GE.ge m (f n) → LT.lt (Dist.dist (u m) (u (f n))) (HPow.h …
    ⊢ Summable fun i => Dist.dist (u (f (HAdd.hAdd i 1))) (u (f i))
  -/
  refine Summable.of_nonneg_of_le (fun n => by positivity) ?_ summable_geometric_two
  /-
    case intro.intro
    α : Type u_1
    inst✝ : PseudoMetricSpace α
    u : Nat → α
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : ∀ (n m : Nat), GE.ge m (f n) → LT.lt (Dist.dist (u m) (u (f n))) (HPow.h …
    ⊢ ∀ (b : Nat), LE.le (Dist.dist (u (f (HAdd.hAdd b 1))) (u (f b))) (HPow.hPow  …
  -/
  exact fun n => le_of_lt <| hf₂ n (f (n+1)) <| hf₁.monotone (Nat.le_add_right n 1)
  /-
    🎉 no goals
  -/


/-- A normed additive group is complete if any absolutely convergent series converges in the
space. -/
lemma NormedAddCommGroup.completeSpace_of_summable_imp_tendsto
    (h : ∀ u : ℕ → E,
      Summable (‖u ·‖) → ∃ a, Tendsto (fun n => ∑ i ∈ range n, u i) atTop (𝓝 a)) :
    CompleteSpace E := by
  /-
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    ⊢ CompleteSpace E
  -/
  apply Metric.complete_of_cauchySeq_tendsto
  /-
    case a
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    ⊢ ∀ (u : Nat → E), CauchySeq u → Exists fun a => Filter.Tendsto u Filter.atTop …
  -/
  intro u hu
  /-
    case a
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    u : Nat → E
    hu : CauchySeq u
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  obtain ⟨f, hf₁, hf₂⟩ := Metric.exists_subseq_summable_dist_of_cauchySeq u hu
  /-
    case a.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    u : Nat → E
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : Summable fun i => Dist.dist (u (f (HAdd.hAdd i 1))) (u (f i))
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  simp only [dist_eq_norm] at hf₂
  /-
    case a.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    u : Nat → E
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : Summable fun i => Norm.norm (HSub.hSub (u (f (HAdd.hAdd i 1))) (u (f i)))
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  let v n := u (f (n+1)) - u (f n)
  have hv_sum : (fun n => (∑ i ∈ range n, v i)) = fun n => u (f n) - u (f 0) := by
    ext n
    exact sum_range_sub (u ∘ f) n
  /-
    case a.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    u : Nat → E
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : Summable fun i => Norm.norm (HSub.hSub (u (f (HAdd.hAdd i 1))) (u (f i)))
    v : Nat → E := fun n => HSub.hSub (u (f (HAdd.hAdd n 1))) (u (f n))
    hv_sum : Eq (fun n => (Finset.range n).sum fun i => v i) fun n => HSub.hSub (u …
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  obtain ⟨a, ha⟩ := h v hf₂
  /-
    case a.intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    u : Nat → E
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : Summable fun i => Norm.norm (HSub.hSub (u (f (HAdd.hAdd i 1))) (u (f i)))
    v : Nat → E := fun n => HSub.hSub (u (f (HAdd.hAdd n 1))) (u (f n))
    hv_sum : Eq (fun n => (Finset.range n).sum fun i => v i) fun n => HSub.hSub (u …
    a : E
    ha : Filter.Tendsto (fun n => (Finset.range n).sum fun i => v i) Filter.atTop  …
    ⊢ Exists fun a => Filter.Tendsto u Filter.atTop (nhds a)
  -/
  refine ⟨a + u (f 0), ?_⟩
  /-
    case a.intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    u : Nat → E
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : Summable fun i => Norm.norm (HSub.hSub (u (f (HAdd.hAdd i 1))) (u (f i)))
    v : Nat → E := fun n => HSub.hSub (u (f (HAdd.hAdd n 1))) (u (f n))
    hv_sum : Eq (fun n => (Finset.range n).sum fun i => v i) fun n => HSub.hSub (u …
    a : E
    ha : Filter.Tendsto (fun n => (Finset.range n).sum fun i => v i) Filter.atTop  …
    ⊢ Filter.Tendsto u Filter.atTop (nhds (HAdd.hAdd a (u (f 0))))
  -/
  refine tendsto_nhds_of_cauchySeq_of_subseq hu hf₁.tendsto_atTop ?_
  /-
    case a.intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    u : Nat → E
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : Summable fun i => Norm.norm (HSub.hSub (u (f (HAdd.hAdd i 1))) (u (f i)))
    v : Nat → E := fun n => HSub.hSub (u (f (HAdd.hAdd n 1))) (u (f n))
    hv_sum : Eq (fun n => (Finset.range n).sum fun i => v i) fun n => HSub.hSub (u …
    a : E
    ha : Filter.Tendsto (fun n => (Finset.range n).sum fun i => v i) Filter.atTop  …
    ⊢ Filter.Tendsto (Function.comp u f) Filter.atTop (nhds (HAdd.hAdd a (u (f 0))))
  -/
  rw [hv_sum] at ha
  have h₁ : Tendsto (fun n => u (f n) - u (f 0) + u (f 0)) atTop (𝓝 (a + u (f 0))) :=
    Tendsto.add_const _ ha
  /-
    case a.intro.intro.intro
    E : Type u_1
    inst✝ : NormedAddCommGroup E
    h : ∀ (u : Nat → E), (Summable fun x => Norm.norm (u x)) → Exists fun a => Fil …
    u : Nat → E
    hu : CauchySeq u
    f : Nat → Nat
    hf₁ : StrictMono f
    hf₂ : Summable fun i => Norm.norm (HSub.hSub (u (f (HAdd.hAdd i 1))) (u (f i)))
    v : Nat → E := fun n => HSub.hSub (u (f (HAdd.hAdd n 1))) (u (f n))
    hv_sum : Eq (fun n => (Finset.range n).sum fun i => v i) fun n => HSub.hSub (u …
    a : E
    ha : Filter.Tendsto (fun n => HSub.hSub (u (f n)) (u (f 0))) Filter.atTop (nhd …
    h₁ : Filter.Tendsto (fun n => HAdd.hAdd (HSub.hSub (u (f n)) (u (f 0))) (u (f  …
    ⊢ Filter.Tendsto (Function.comp u f) Filter.atTop (nhds (HAdd.hAdd a (u (f 0))))
  -/
  simpa only [sub_add_cancel] using h₁
  /-
    🎉 no goals
  -/


/-- In a complete normed additive group, every absolutely convergent series converges in the
space. -/
lemma NormedAddCommGroup.summable_imp_tendsto_of_complete [CompleteSpace E] (u : ℕ → E)
    (hu : Summable (‖u ·‖)) : ∃ a, Tendsto (fun n => ∑ i ∈ range n, u i) atTop (𝓝 a) := by
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    u : Nat → E
    hu : Summable fun x => Norm.norm (u x)
    ⊢ Exists fun a => Filter.Tendsto (fun n => (Finset.range n).sum fun i => u i)  …
  -/
  refine cauchySeq_tendsto_of_complete <| cauchySeq_of_summable_dist ?_
  /-
    E : Type u_1
    inst✝¹ : NormedAddCommGroup E
    inst✝ : CompleteSpace E
    u : Nat → E
    hu : Summable fun x => Norm.norm (u x)
    ⊢ Summable fun n => Dist.dist ((Finset.range n).sum fun i => u i) ((Finset.ran …
  -/
  simp [dist_eq_norm, sum_range_succ, hu]
  /-
    🎉 no goals
  -/


/-- In a normed additive group, every absolutely convergent series converges in the
space iff the space is complete. -/
lemma NormedAddCommGroup.summable_imp_tendsto_iff_completeSpace :
    (∀ u : ℕ → E, Summable (‖u ·‖) → ∃ a, Tendsto (fun n => ∑ i ∈ range n, u i) atTop (𝓝 a))
     ↔ CompleteSpace E :=
  ⟨completeSpace_of_summable_imp_tendsto, fun _ u hu => summable_imp_tendsto_of_complete u hu⟩


