@[to_additive]
theorem exists_finsupp_of_mem_closure_range (hx : x ∈ closure (Set.range f)) :
    ∃ a : ι →₀ ℕ, x = a.prod (f · ^ ·) := by
  classical
  induction hx using closure_induction with
  | mem x h => obtain ⟨i, rfl⟩ := h; exact ⟨Finsupp.single i 1, by simp⟩
  | one => use 0; simp
  | mul x y hx hy hx' hy' =>
    obtain ⟨⟨v, rfl⟩, w, rfl⟩ := And.intro hx' hy'
    use v + w
    rw [Finsupp.prod_add_index]
    · simp
    · simp [pow_add]


variable {f x} in
@[to_additive]
theorem mem_closure_range_iff :
    x ∈ closure (Set.range f) ↔ ∃ a : ι →₀ ℕ, x = a.prod (f · ^ ·) := by
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    ι : Type u_2
    f : ι → M
    x : M
    ⊢ Iff (Membership.mem (Submonoid.closure (Set.range f)) x) (Exists fun a => Eq …
  -/
  refine ⟨exists_finsupp_of_mem_closure_range f x, ?_⟩
  /-
    M : Type u_1
    inst✝ : CommMonoid M
    ι : Type u_2
    f : ι → M
    x : M
    ⊢ (Exists fun a => Eq x (a.prod fun x1 x2 => HPow.hPow (f x1) x2)) → Membershi …
  -/
  rintro ⟨a, rfl⟩
  /-
    case intro
    M : Type u_1
    inst✝ : CommMonoid M
    ι : Type u_2
    f : ι → M
    a : Finsupp ι Nat
    ⊢ Membership.mem (Submonoid.closure (Set.range f)) (a.prod fun x1 x2 => HPow.h …
  -/
  exact prod_mem _ fun i hi ↦ pow_mem (subset_closure (Set.mem_range_self i)) _
  /-
    🎉 no goals
  -/


@[to_additive]
theorem exists_of_mem_closure_range [Fintype ι] (hx : x ∈ closure (Set.range f)) :
    ∃ a : ι → ℕ, x = ∏ i, f i ^ a i := by
  /-
    M : Type u_1
    inst✝¹ : CommMonoid M
    ι : Type u_2
    f : ι → M
    x : M
    inst✝ : Fintype ι
    hx : Membership.mem (Submonoid.closure (Set.range f)) x
    ⊢ Exists fun a => Eq x (Finset.univ.prod fun i => HPow.hPow (f i) (a i))
  -/
  obtain ⟨a, rfl⟩ := exists_finsupp_of_mem_closure_range f x hx
  /-
    case intro
    M : Type u_1
    inst✝¹ : CommMonoid M
    ι : Type u_2
    f : ι → M
    inst✝ : Fintype ι
    a : Finsupp ι Nat
    hx : Membership.mem (Submonoid.closure (Set.range f)) (a.prod fun x1 x2 => HPo …
    ⊢ Exists fun a_1 => Eq (a.prod fun x1 x2 => HPow.hPow (f x1) x2) (Finset.univ. …
  -/
  exact ⟨a, by simp⟩
  /-
    🎉 no goals
  -/


variable {f x} in
@[to_additive]
theorem mem_closure_range_iff_of_fintype [Fintype ι] :
    x ∈ closure (Set.range f) ↔ ∃ a : ι → ℕ, x = ∏ i, f i ^ a i := by
  /-
    M : Type u_1
    inst✝¹ : CommMonoid M
    ι : Type u_2
    f : ι → M
    x : M
    inst✝ : Fintype ι
    ⊢ Iff (Membership.mem (Submonoid.closure (Set.range f)) x) (Exists fun a => Eq …
  -/
  rw [Finsupp.equivFunOnFinite.symm.exists_congr_left, mem_closure_range_iff]
  /-
    M : Type u_1
    inst✝¹ : CommMonoid M
    ι : Type u_2
    f : ι → M
    x : M
    inst✝ : Fintype ι
    ⊢ Iff (Exists fun a => Eq x (a.prod fun x1 x2 => HPow.hPow (f x1) x2)) (Exists …
  -/
  simp
  /-
    🎉 no goals
  -/


