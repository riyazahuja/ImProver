/-- Create an element of `ι →₀ α` from a finset `s` and a function `f` defined on this finset. -/
def indicator (s : Finset ι) (f : ∀ i ∈ s, α) : ι →₀ α where
  toFun i :=
    haveI := Classical.decEq ι
    if H : i ∈ s then f i H else 0
  support :=
    haveI := Classical.decEq α
    ({i | f i.1 i.2 ≠ 0} : Finset s).map (Embedding.subtype _)
  mem_support_toFun i := by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝ : Zero α
      s✝ : Finset ι
      f✝ : (i : ι) → Membership.mem s✝ i → α
      i✝ : ι
      s : Finset ι
      f : (i : ι) → Membership.mem s i → α
      i : ι
      ⊢ Iff (Membership.mem (Finset.map (Function.Embedding.subtype fun x => Members …
    -/
    classical simp
    /-
      🎉 no goals
    -/


theorem indicator_of_mem (hi : i ∈ s) (f : ∀ i ∈ s, α) : indicator s f i = f i hi :=
  @dif_pos _ (id _) hi _ _ _


theorem indicator_of_not_mem (hi : i ∉ s) (f : ∀ i ∈ s, α) : indicator s f i = 0 :=
  @dif_neg _ (id _) hi _ _ _


@[simp]
theorem indicator_apply [DecidableEq ι] : indicator s f i = if hi : i ∈ s then f i hi else 0 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : Zero α
    s : Finset ι
    f : (i : ι) → Membership.mem s i → α
    i : ι
    inst✝ : DecidableEq ι
    ⊢ Eq ((Finsupp.indicator s f) i) (dite (Membership.mem s i) (fun hi => f i hi) …
  -/
  simp only [indicator, ne_eq, coe_mk]
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : Zero α
    s : Finset ι
    f : (i : ι) → Membership.mem s i → α
    i : ι
    inst✝ : DecidableEq ι
    ⊢ Eq (dite (Membership.mem s i) (fun H => f i H) fun H => 0) (dite (Membership …
  -/
  congr
  /-
    🎉 no goals
  -/


theorem indicator_injective : Injective fun f : ∀ i ∈ s, α => indicator s f := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    ⊢ Function.Injective fun f => Finsupp.indicator s f
  -/
  intro a b h
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    a b : (i : ι) → Membership.mem s i → α
    h : Eq ((fun f => Finsupp.indicator s f) a) ((fun f => Finsupp.indicator s f) b)
    ⊢ Eq a b
  -/
  ext i hi
  /-
    case h.h
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    a b : (i : ι) → Membership.mem s i → α
    h : Eq ((fun f => Finsupp.indicator s f) a) ((fun f => Finsupp.indicator s f) b)
    i : ι
    hi : Membership.mem s i
    ⊢ Eq (a i hi) (b i hi)
  -/
  rw [← indicator_of_mem hi a, ← indicator_of_mem hi b]
  /-
    case h.h
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    a b : (i : ι) → Membership.mem s i → α
    h : Eq ((fun f => Finsupp.indicator s f) a) ((fun f => Finsupp.indicator s f) b)
    i : ι
    hi : Membership.mem s i
    ⊢ Eq ((Finsupp.indicator s a) i) ((Finsupp.indicator s b) i)
  -/
  exact DFunLike.congr_fun h i
  /-
    🎉 no goals
  -/


theorem support_indicator_subset : ((indicator s f).support : Set ι) ⊆ s := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    f : (i : ι) → Membership.mem s i → α
    ⊢ HasSubset.Subset ↑(Finsupp.indicator s f).support ↑s
  -/
  intro i hi
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    f : (i : ι) → Membership.mem s i → α
    i : ι
    hi : Membership.mem (↑(Finsupp.indicator s f).support) i
    ⊢ Membership.mem (↑s) i
  -/
  rw [mem_coe, mem_support_iff] at hi
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    f : (i : ι) → Membership.mem s i → α
    i : ι
    hi : Ne ((Finsupp.indicator s f) i) 0
    ⊢ Membership.mem (↑s) i
  -/
  by_contra h
  /-
    ι : Type u_1
    α : Type u_2
    inst✝ : Zero α
    s : Finset ι
    f : (i : ι) → Membership.mem s i → α
    i : ι
    hi : Ne ((Finsupp.indicator s f) i) 0
    h : Not (Membership.mem (↑s) i)
    ⊢ False
  -/
  exact hi (indicator_of_not_mem h _)
  /-
    🎉 no goals
  -/


lemma single_eq_indicator (b : α) : single i b = indicator {i} (fun _ _ => b) := by
  classical
  ext j
  simp [single_apply, indicator_apply, @eq_comm _ j]


