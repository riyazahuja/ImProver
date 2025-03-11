/-- An infinite sum of functions with summable sup norm is the uniform limit of its partial sums.
Version relative to a set, with general index set. -/
theorem tendstoUniformlyOn_tsum {f : α → β → F} (hu : Summable u) {s : Set β}
    (hfu : ∀ n x, x ∈ s → ‖f n x‖ ≤ u n) :
    TendstoUniformlyOn (fun t : Finset α => fun x => ∑ n ∈ t, f n x) (fun x => ∑' n, f n x) atTop
      s := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    u : α → Real
    f : α → β → F
    hu : Summable u
    s : Set β
    hfu : ∀ (n : α) (x : β), Membership.mem s x → LE.le (Norm.norm (f n x)) (u n)
    ⊢ TendstoUniformlyOn (fun t x => t.sum fun n => f n x) (fun x => tsum fun n => …
  -/
  refine tendstoUniformlyOn_iff.2 fun ε εpos => ?_
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    u : α → Real
    f : α → β → F
    hu : Summable u
    s : Set β
    hfu : ∀ (n : α) (x : β), Membership.mem s x → LE.le (Norm.norm (f n x)) (u n)
    ε : Real
    εpos : GT.gt ε 0
    ⊢ Filter.Eventually (fun n => ∀ (x : β), Membership.mem s x → LT.lt (Dist.dist …
  -/
  filter_upwards [(tendsto_order.1 (tendsto_tsum_compl_atTop_zero u)).2 _ εpos] with t ht x hx
  have A : Summable fun n => ‖f n x‖ :=
    .of_nonneg_of_le (fun _ ↦ norm_nonneg _) (fun n => hfu n x hx) hu
  /-
    case h
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    u : α → Real
    f : α → β → F
    hu : Summable u
    s : Set β
    hfu : ∀ (n : α) (x : β), Membership.mem s x → LE.le (Norm.norm (f n x)) (u n)
    ε : Real
    εpos : GT.gt ε 0
    t : Finset α
    ht : LT.lt (tsum fun a => u ↑a) ε
    x : β
    hx : Membership.mem s x
    A : Summable fun n => Norm.norm (f n x)
    ⊢ LT.lt (Dist.dist (tsum fun n => f n x) (t.sum fun n => f n x)) ε
  -/
  rw [dist_eq_norm, ← sum_add_tsum_subtype_compl A.of_norm t, add_sub_cancel_left]
  /-
    case h
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    u : α → Real
    f : α → β → F
    hu : Summable u
    s : Set β
    hfu : ∀ (n : α) (x : β), Membership.mem s x → LE.le (Norm.norm (f n x)) (u n)
    ε : Real
    εpos : GT.gt ε 0
    t : Finset α
    ht : LT.lt (tsum fun a => u ↑a) ε
    x : β
    hx : Membership.mem s x
    A : Summable fun n => Norm.norm (f n x)
    ⊢ LT.lt (Norm.norm (tsum fun x_1 => f (↑x_1) x)) ε
  -/
  apply lt_of_le_of_lt _ ht
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    u : α → Real
    f : α → β → F
    hu : Summable u
    s : Set β
    hfu : ∀ (n : α) (x : β), Membership.mem s x → LE.le (Norm.norm (f n x)) (u n)
    ε : Real
    εpos : GT.gt ε 0
    t : Finset α
    ht : LT.lt (tsum fun a => u ↑a) ε
    x : β
    hx : Membership.mem s x
    A : Summable fun n => Norm.norm (f n x)
    ⊢ LE.le (Norm.norm (tsum fun x_1 => f (↑x_1) x)) (tsum fun a => u ↑a)
  -/
  apply (norm_tsum_le_tsum_norm (A.subtype _)).trans
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    u : α → Real
    f : α → β → F
    hu : Summable u
    s : Set β
    hfu : ∀ (n : α) (x : β), Membership.mem s x → LE.le (Norm.norm (f n x)) (u n)
    ε : Real
    εpos : GT.gt ε 0
    t : Finset α
    ht : LT.lt (tsum fun a => u ↑a) ε
    x : β
    hx : Membership.mem s x
    A : Summable fun n => Norm.norm (f n x)
    ⊢ LE.le (tsum fun i => Norm.norm (f (↑i) x)) (tsum fun a => u ↑a)
  -/
  exact tsum_le_tsum (fun n => hfu _ _ hx) (A.subtype _) (hu.subtype _)
  /-
    🎉 no goals
  -/


/-- An infinite sum of functions with summable sup norm is the uniform limit of its partial sums.
Version relative to a set, with index set `ℕ`. -/
theorem tendstoUniformlyOn_tsum_nat {f : ℕ → β → F} {u : ℕ → ℝ} (hu : Summable u) {s : Set β}
    (hfu : ∀ n x, x ∈ s → ‖f n x‖ ≤ u n) :
    TendstoUniformlyOn (fun N => fun x => ∑ n ∈ Finset.range N, f n x) (fun x => ∑' n, f n x) atTop
      s :=
  fun v hv => tendsto_finset_range.eventually (tendstoUniformlyOn_tsum hu hfu v hv)


/-- An infinite sum of functions with eventually summable sup norm is the uniform limit of its
partial sums. Version relative to a set, with general index set. -/
theorem tendstoUniformlyOn_tsum_of_cofinite_eventually {ι : Type*} {f : ι → β → F} {u : ι → ℝ}
    (hu : Summable u) {s : Set β} (hfu : ∀ᶠ n in cofinite, ∀ x ∈ s, ‖f n x‖ ≤ u n) :
    TendstoUniformlyOn (fun t x => ∑ n ∈ t, f n x) (fun x => ∑' n, f n x) atTop s := by
  classical
  refine tendstoUniformlyOn_iff.2 fun ε εpos => ?_
  have := (tendsto_order.1 (tendsto_tsum_compl_atTop_zero u)).2 _ εpos
  simp only [not_forall, Classical.not_imp, not_le, gt_iff_lt,
    eventually_atTop, ge_iff_le, Finset.le_eq_subset] at *
  obtain ⟨t, ht⟩ := this
  rw [eventually_iff_exists_mem] at hfu
  obtain ⟨N, hN, HN⟩ := hfu
  refine ⟨hN.toFinset ∪ t, fun n hn x hx => ?_⟩
  have A : Summable fun n => ‖f n x‖ := by
    apply Summable.add_compl (s := hN.toFinset) Summable.of_finite
    apply Summable.of_nonneg_of_le (fun _ ↦ norm_nonneg _) _ (hu.subtype _)
    simp only [comp_apply, Subtype.forall, Set.mem_compl_iff, Finset.mem_coe]
    aesop
  rw [dist_eq_norm, ← sum_add_tsum_subtype_compl A.of_norm n, add_sub_cancel_left]
  apply lt_of_le_of_lt _ (ht n (Finset.union_subset_right hn))
  apply (norm_tsum_le_tsum_norm (A.subtype _)).trans
  apply tsum_le_tsum _ (A.subtype _) (hu.subtype _)
  simp only [comp_apply, Subtype.forall, imp_false]
  apply fun i hi => HN i ?_ x hx
  have : ¬ i ∈ hN.toFinset := fun hg ↦ hi (Finset.union_subset_left hn hg)
  aesop


/-- An infinite sum of functions with summable sup norm is the uniform limit of its partial sums.
Version with general index set. -/
theorem tendstoUniformly_tsum {f : α → β → F} (hu : Summable u) (hfu : ∀ n x, ‖f n x‖ ≤ u n) :
    TendstoUniformly (fun t : Finset α => fun x => ∑ n ∈ t, f n x)
      (fun x => ∑' n, f n x) atTop := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    u : α → Real
    f : α → β → F
    hu : Summable u
    hfu : ∀ (n : α) (x : β), LE.le (Norm.norm (f n x)) (u n)
    ⊢ TendstoUniformly (fun t x => t.sum fun n => f n x) (fun x => tsum fun n => f …
  -/
  rw [← tendstoUniformlyOn_univ]; exact tendstoUniformlyOn_tsum hu fun n x _ => hfu n x
                                  /-
                                    🎉 no goals
                                  -/


/-- An infinite sum of functions with summable sup norm is the uniform limit of its partial sums.
Version with index set `ℕ`. -/
theorem tendstoUniformly_tsum_nat {f : ℕ → β → F} {u : ℕ → ℝ} (hu : Summable u)
    (hfu : ∀ n x, ‖f n x‖ ≤ u n) :
    TendstoUniformly (fun N => fun x => ∑ n ∈ Finset.range N, f n x) (fun x => ∑' n, f n x)
      atTop :=
  fun v hv => tendsto_finset_range.eventually (tendstoUniformly_tsum hu hfu v hv)


/-- An infinite sum of functions with eventually summable sup norm is the uniform limit of its
partial sums. Version with general index set. -/
theorem tendstoUniformly_tsum_of_cofinite_eventually {ι : Type*} {f : ι → β → F} {u : ι → ℝ}
    (hu : Summable u) (hfu : ∀ᶠ (n : ι) in cofinite, ∀ x : β, ‖f n x‖ ≤ u n) :
    TendstoUniformly (fun t x => ∑ n ∈ t, f n x) (fun x => ∑' n, f n x) atTop := by
  /-
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    ι : Type u_4
    f : ι → β → F
    u : ι → Real
    hu : Summable u
    hfu : Filter.Eventually (fun n => ∀ (x : β), LE.le (Norm.norm (f n x)) (u n))  …
    ⊢ TendstoUniformly (fun t x => t.sum fun n => f n x) (fun x => tsum fun n => f …
  -/
  rw [← tendstoUniformlyOn_univ]
  /-
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    ι : Type u_4
    f : ι → β → F
    u : ι → Real
    hu : Summable u
    hfu : Filter.Eventually (fun n => ∀ (x : β), LE.le (Norm.norm (f n x)) (u n))  …
    ⊢ TendstoUniformlyOn (fun t x => t.sum fun n => f n x) (fun x => tsum fun n => …
  -/
  apply tendstoUniformlyOn_tsum_of_cofinite_eventually hu
  /-
    β : Type u_2
    F : Type u_3
    inst✝¹ : NormedAddCommGroup F
    inst✝ : CompleteSpace F
    ι : Type u_4
    f : ι → β → F
    u : ι → Real
    hu : Summable u
    hfu : Filter.Eventually (fun n => ∀ (x : β), LE.le (Norm.norm (f n x)) (u n))  …
    ⊢ Filter.Eventually (fun n => ∀ (x : β), Membership.mem Set.univ x → LE.le (No …
  -/
  simpa using hfu
  /-
    🎉 no goals
  -/


/-- An infinite sum of functions with summable sup norm is continuous on a set if each individual
function is. -/
theorem continuousOn_tsum [TopologicalSpace β] {f : α → β → F} {s : Set β}
    (hf : ∀ i, ContinuousOn (f i) s) (hu : Summable u) (hfu : ∀ n x, x ∈ s → ‖f n x‖ ≤ u n) :
    ContinuousOn (fun x => ∑' n, f n x) s := by
  classical
    refine (tendstoUniformlyOn_tsum hu hfu).continuousOn (Eventually.of_forall ?_)
    intro t
    exact continuousOn_finset_sum _ fun i _ => hf i


/-- An infinite sum of functions with summable sup norm is continuous if each individual
function is. -/
theorem continuous_tsum [TopologicalSpace β] {f : α → β → F} (hf : ∀ i, Continuous (f i))
    (hu : Summable u) (hfu : ∀ n x, ‖f n x‖ ≤ u n) : Continuous fun x => ∑' n, f n x := by
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : TopologicalSpace β
    f : α → β → F
    hf : ∀ (i : α), Continuous (f i)
    hu : Summable u
    hfu : ∀ (n : α) (x : β), LE.le (Norm.norm (f n x)) (u n)
    ⊢ Continuous fun x => tsum fun n => f n x
  -/
  simp_rw [continuous_iff_continuousOn_univ] at hf ⊢
  /-
    α : Type u_1
    β : Type u_2
    F : Type u_3
    inst✝² : NormedAddCommGroup F
    inst✝¹ : CompleteSpace F
    u : α → Real
    inst✝ : TopologicalSpace β
    f : α → β → F
    hu : Summable u
    hfu : ∀ (n : α) (x : β), LE.le (Norm.norm (f n x)) (u n)
    hf : ∀ (i : α), ContinuousOn (f i) Set.univ
    ⊢ ContinuousOn (fun x => tsum fun n => f n x) Set.univ
  -/
  exact continuousOn_tsum hf hu fun n x _ => hfu n x
  /-
    🎉 no goals
  -/

