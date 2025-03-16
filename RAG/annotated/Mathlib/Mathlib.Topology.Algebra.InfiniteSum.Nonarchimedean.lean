/-- Let `G` be a nonarchimedean multiplicative abelian group, and let `f : α → G` be a function that
tends to one on the filter of cofinite sets. For each finite subset of `α`, consider the partial
product of `f` on that subset. These partial products form a Cauchy filter. -/
@[to_additive "Let `G` be a nonarchimedean additive abelian group, and let `f : α → G` be a function
that tends to zero on the filter of cofinite sets. For each finite subset of `α`, consider the
partial sum of `f` on that subset. These partial sums form a Cauchy filter."]
theorem cauchySeq_prod_of_tendsto_cofinite_one {f : α → G} (hf : Tendsto f cofinite (𝓝 1)) :
    CauchySeq (fun s ↦ ∏ i ∈ s, f i) := by
  /- Let `U` be a neighborhood of `1`. It suffices to show that there exists `s : Finset α` such
  that for any `t : Finset α` disjoint from `s`, we have `∏ i ∈ t, f i ∈ U`. -/
  /-
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    ⊢ CauchySeq fun s => s.prod fun i => f i
  -/
  apply cauchySeq_finset_iff_prod_vanishing.mpr
  /-
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    ⊢ ∀ (e : Set G), Membership.mem (nhds 1) e → Exists fun s => ∀ (t : Finset α), …
  -/
  intro U hU
  -- Since `G` is nonarchimedean, `U` contains an open subgroup `V`.
  /-
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    U : Set G
    hU : Membership.mem (nhds 1) U
    ⊢ Exists fun s => ∀ (t : Finset α), Disjoint t s → Membership.mem U (t.prod fu …
  -/
  rcases is_nonarchimedean U hU with ⟨V, hV⟩
  /- Let `s` be the set of all indices `i : α` such that `f i ∉ V`. By our assumption `hf`, this is
  finite. -/
  /-
    case intro
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    U : Set G
    hU : Membership.mem (nhds 1) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U
    ⊢ Exists fun s => ∀ (t : Finset α), Disjoint t s → Membership.mem U (t.prod fu …
  -/
  use (tendsto_def.mp hf V V.mem_nhds_one).toFinset
  /- For any `t : Finset α` disjoint from `s`, the product `∏ i ∈ t, f i` is a product of elements
  of `V`, so it is an element of `V` too. Thus, `∏ i ∈ t, f i ∈ U`, as desired. -/
  /-
    case h
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    U : Set G
    hU : Membership.mem (nhds 1) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U
    ⊢ ∀ (t : Finset α), Disjoint t (Set.Finite.toFinset ⋯) → Membership.mem U (t.p …
  -/
  intro t ht
  /-
    case h
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    U : Set G
    hU : Membership.mem (nhds 1) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U
    t : Finset α
    ht : Disjoint t (Set.Finite.toFinset ⋯)
    ⊢ Membership.mem U (t.prod fun b => f b)
  -/
  apply hV
  /-
    case h.a
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    U : Set G
    hU : Membership.mem (nhds 1) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U
    t : Finset α
    ht : Disjoint t (Set.Finite.toFinset ⋯)
    ⊢ Membership.mem (↑V) (t.prod fun b => f b)
  -/
  apply Subgroup.prod_mem
  /-
    case h.a.h
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    U : Set G
    hU : Membership.mem (nhds 1) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U
    t : Finset α
    ht : Disjoint t (Set.Finite.toFinset ⋯)
    ⊢ ∀ (c : α), Membership.mem t c → Membership.mem (↑V) (f c)
  -/
  intro i hi
  /-
    case h.a.h
    α : Type u_1
    G : Type u_2
    inst✝³ : CommGroup G
    inst✝² : UniformSpace G
    inst✝¹ : UniformGroup G
    inst✝ : NonarchimedeanGroup G
    f : α → G
    hf : Filter.Tendsto f Filter.cofinite (nhds 1)
    U : Set G
    hU : Membership.mem (nhds 1) U
    V : OpenSubgroup G
    hV : HasSubset.Subset (↑V) U
    t : Finset α
    ht : Disjoint t (Set.Finite.toFinset ⋯)
    i : α
    hi : Membership.mem t i
    ⊢ Membership.mem (↑V) (f i)
  -/
  simpa using Finset.disjoint_left.mp ht hi
  /-
    🎉 no goals
  -/


/-- Let `G` be a complete nonarchimedean multiplicative abelian group, and let `f : α → G` be a
function that tends to one on the filter of cofinite sets. Then `f` is unconditionally
multipliable. -/
@[to_additive "Let `G` be a complete nonarchimedean additive abelian group, and let `f : α → G` be a
function that tends to zero on the filter of cofinite sets. Then `f` is unconditionally summable."]
theorem multipliable_of_tendsto_cofinite_one [CompleteSpace G] {f : α → G}
    (hf : Tendsto f cofinite (𝓝 1)) : Multipliable f :=
  CompleteSpace.complete (cauchySeq_prod_of_tendsto_cofinite_one hf)


/-- Let `G` be a complete nonarchimedean multiplicative abelian group. Then a function `f : α → G`
is unconditionally multipliable if and only if it tends to one on the filter of cofinite sets. -/
@[to_additive "Let `G` be a complete nonarchimedean additive abelian group. Then a function
`f : α → G` is unconditionally summable if and only if it tends to zero on the filter of cofinite
sets."]
theorem multipliable_iff_tendsto_cofinite_one [CompleteSpace G] (f : α → G) :
    Multipliable f ↔ Tendsto f cofinite (𝓝 1) :=
  ⟨Multipliable.tendsto_cofinite_one, multipliable_of_tendsto_cofinite_one⟩


private theorem Summable.mul_of_complete_nonarchimedean [CompleteSpace R] {f : α → R} {g : β → R}
    (hf : Summable f) (hg : Summable g) : Summable (fun i : α × β ↦ f i.1 * g i.2) := by
  /-
    α : Type u_1
    β : Type u_2
    R : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : UniformSpace R
    inst✝² : UniformAddGroup R
    inst✝¹ : NonarchimedeanRing R
    inst✝ : CompleteSpace R
    f : α → R
    g : β → R
    hf : Summable f
    hg : Summable g
    ⊢ Summable fun i => HMul.hMul (f i.1) (g i.2)
  -/
  rw [NonarchimedeanAddGroup.summable_iff_tendsto_cofinite_zero] at *
  /-
    α : Type u_1
    β : Type u_2
    R : Type u_3
    inst✝⁴ : Ring R
    inst✝³ : UniformSpace R
    inst✝² : UniformAddGroup R
    inst✝¹ : NonarchimedeanRing R
    inst✝ : CompleteSpace R
    f : α → R
    g : β → R
    hf : Filter.Tendsto f Filter.cofinite (nhds 0)
    hg : Filter.Tendsto g Filter.cofinite (nhds 0)
    ⊢ Filter.Tendsto (fun i => HMul.hMul (f i.1) (g i.2)) Filter.cofinite (nhds 0)
  -/
  exact tendsto_mul_cofinite_nhds_zero hf hg
  /-
    🎉 no goals
  -/


/-- Let `R` be a nonarchimedean ring, let `f : α → R` be a function that sums to `a : R`,
and let `g : β → R` be a function that sums to `b : R`. Then `fun i : α × β ↦ f i.1 * g i.2`
sums to `a * b`. -/
theorem HasSum.mul_of_nonarchimedean {f : α → R} {g : β → R} {a b : R} (hf : HasSum f a)
    (hg : HasSum g b) : HasSum (fun i : α × β ↦ f i.1 * g i.2) (a * b) := by
  /-
    α : Type u_1
    β : Type u_2
    R : Type u_3
    inst✝³ : Ring R
    inst✝² : UniformSpace R
    inst✝¹ : UniformAddGroup R
    inst✝ : NonarchimedeanRing R
    f : α → R
    g : β → R
    a b : R
    hf : HasSum f a
    hg : HasSum g b
    ⊢ HasSum (fun i => HMul.hMul (f i.1) (g i.2)) (HMul.hMul a b)
  -/
  rw [← hasSum_iff_hasSum_compl] at *
  simp only [Function.comp_def, UniformSpace.Completion.toCompl_apply,
    UniformSpace.Completion.coe_mul]
  /-
    α : Type u_1
    β : Type u_2
    R : Type u_3
    inst✝³ : Ring R
    inst✝² : UniformSpace R
    inst✝¹ : UniformAddGroup R
    inst✝ : NonarchimedeanRing R
    f : α → R
    g : β → R
    a b : R
    hf : HasSum (Function.comp (⇑UniformSpace.Completion.toCompl) f) (↑R a)
    hg : HasSum (Function.comp (⇑UniformSpace.Completion.toCompl) g) (↑R b)
    ⊢ HasSum (fun x => HMul.hMul (↑R (f x.1)) (↑R (g x.2))) (HMul.hMul (↑R a) (↑R  …
  -/
  exact (hf.mul hg) (hf.summable.mul_of_complete_nonarchimedean hg.summable :)
  /-
    🎉 no goals
  -/


/-- Let `R` be a nonarchimedean ring. If functions `f : α → R` and `g : β → R` are summable, then
so is `fun i : α × β ↦ f i.1 * g i.2`. -/
theorem Summable.mul_of_nonarchimedean {f : α → R} {g : β → R} (hf : Summable f)
    (hg : Summable g) : Summable (fun i : α × β ↦ f i.1 * g i.2) :=
  (hf.hasSum.mul_of_nonarchimedean hg.hasSum).summable


theorem tsum_mul_tsum_of_nonarchimedean [T0Space R] {f : α → R} {g : β → R} (hf : Summable f)
    (hg : Summable g) : (∑' i, f i) * (∑' i, g i) = ∑' i : α × β, f i.1 * g i.2 :=
  (hf.hasSum.mul_of_nonarchimedean hg.hasSum).tsum_eq.symm


