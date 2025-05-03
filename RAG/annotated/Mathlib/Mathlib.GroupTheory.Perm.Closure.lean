theorem closure_isCycle : closure { σ : Perm β | IsCycle σ } = ⊤ := by
  classical
    cases nonempty_fintype β
    exact
      top_le_iff.mp (le_trans (ge_of_eq closure_isSwap) (closure_mono fun _ => IsSwap.isCycle))


theorem closure_cycle_adjacent_swap {σ : Perm α} (h1 : IsCycle σ) (h2 : σ.support = univ) (x : α) :
    closure ({σ, swap x (σ x)} : Set (Perm α)) = ⊤ := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x (σ  …
  -/
  let H := closure ({σ, swap x (σ x)} : Set (Perm α))
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    H : Subgroup (Equiv.Perm α) := Subgroup.closure (Insert.insert σ (Singleton.si …
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x (σ  …
  -/
  have h3 : σ ∈ H := subset_closure (Set.mem_insert σ _)
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    H : Subgroup (Equiv.Perm α) := Subgroup.closure (Insert.insert σ (Singleton.si …
    h3 : Membership.mem H σ
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x (σ  …
  -/
  have h4 : swap x (σ x) ∈ H := subset_closure (Set.mem_insert_of_mem _ (Set.mem_singleton _))
  have step1 : ∀ n : ℕ, swap ((σ ^ n) x) ((σ ^ (n + 1) : Perm α) x) ∈ H := by
    intro n
    induction n with
    | zero => exact subset_closure (Set.mem_insert_of_mem _ (Set.mem_singleton _))
    | succ n ih =>
      convert H.mul_mem (H.mul_mem h3 ih) (H.inv_mem h3)
      simp_rw [mul_swap_eq_swap_mul, mul_inv_cancel_right, pow_succ', coe_mul, comp_apply]
  have step2 : ∀ n : ℕ, swap x ((σ ^ n) x) ∈ H := by
    intro n
    induction n with
    | zero =>
      simp only [pow_zero, coe_one, id_eq, swap_self, Set.mem_singleton_iff]
      convert H.one_mem
    | succ n ih =>
      by_cases h5 : x = (σ ^ n) x
      · rw [pow_succ', mul_apply, ← h5]
        exact h4
      by_cases h6 : x = (σ ^ (n + 1) : Perm α) x
      · rw [← h6, swap_self]
        exact H.one_mem
      rw [swap_comm, ← swap_mul_swap_mul_swap h5 h6]
      exact H.mul_mem (H.mul_mem (step1 n) ih) (step1 n)
  have step3 : ∀ y : α, swap x y ∈ H := by
    intro y
    have hx : x ∈ univ := Finset.mem_univ x
    rw [← h2, mem_support] at hx
    have hy : y ∈ univ := Finset.mem_univ y
    rw [← h2, mem_support] at hy
    cases' IsCycle.exists_pow_eq h1 hx hy with n hn
    rw [← hn]
    exact step2 n
  have step4 : ∀ y z : α, swap y z ∈ H := by
    intro y z
    by_cases h5 : z = x
    · rw [h5, swap_comm]
      exact step3 y
    by_cases h6 : z = y
    · rw [h6, swap_self]
      exact H.one_mem
    rw [← swap_mul_swap_mul_swap h5 h6, swap_comm z x]
    exact H.mul_mem (H.mul_mem (step3 y) (step3 z)) (step3 y)
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    H : Subgroup (Equiv.Perm α) := Subgroup.closure (Insert.insert σ (Singleton.si …
    h3 : Membership.mem H σ
    h4 : Membership.mem H (Equiv.swap x (σ x))
    step1 : ∀ (n : Nat), Membership.mem H (Equiv.swap ((HPow.hPow σ n) x) ((HPow.h …
    step2 : ∀ (n : Nat), Membership.mem H (Equiv.swap x ((HPow.hPow σ n) x))
    step3 : ∀ (y : α), Membership.mem H (Equiv.swap x y)
    step4 : ∀ (y z : α), Membership.mem H (Equiv.swap y z)
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x (σ  …
  -/
  rw [eq_top_iff, ← closure_isSwap, closure_le]
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    H : Subgroup (Equiv.Perm α) := Subgroup.closure (Insert.insert σ (Singleton.si …
    h3 : Membership.mem H σ
    h4 : Membership.mem H (Equiv.swap x (σ x))
    step1 : ∀ (n : Nat), Membership.mem H (Equiv.swap ((HPow.hPow σ n) x) ((HPow.h …
    step2 : ∀ (n : Nat), Membership.mem H (Equiv.swap x ((HPow.hPow σ n) x))
    step3 : ∀ (y : α), Membership.mem H (Equiv.swap x y)
    step4 : ∀ (y z : α), Membership.mem H (Equiv.swap y z)
    ⊢ HasSubset.Subset (setOf fun σ => σ.IsSwap) ↑(Subgroup.closure (Insert.insert …
  -/
  rintro τ ⟨y, z, _, h6⟩
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    H : Subgroup (Equiv.Perm α) := Subgroup.closure (Insert.insert σ (Singleton.si …
    h3 : Membership.mem H σ
    h4 : Membership.mem H (Equiv.swap x (σ x))
    step1 : ∀ (n : Nat), Membership.mem H (Equiv.swap ((HPow.hPow σ n) x) ((HPow.h …
    step2 : ∀ (n : Nat), Membership.mem H (Equiv.swap x ((HPow.hPow σ n) x))
    step3 : ∀ (y : α), Membership.mem H (Equiv.swap x y)
    step4 : ∀ (y z : α), Membership.mem H (Equiv.swap y z)
    τ : Equiv.Perm α
    y z : α
    left✝ : Ne y z
    h6 : Eq τ (Equiv.swap y z)
    ⊢ Membership.mem (↑(Subgroup.closure (Insert.insert σ (Singleton.singleton (Eq …
  -/
  rw [h6]
  /-
    case intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ : Equiv.Perm α
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    H : Subgroup (Equiv.Perm α) := Subgroup.closure (Insert.insert σ (Singleton.si …
    h3 : Membership.mem H σ
    h4 : Membership.mem H (Equiv.swap x (σ x))
    step1 : ∀ (n : Nat), Membership.mem H (Equiv.swap ((HPow.hPow σ n) x) ((HPow.h …
    step2 : ∀ (n : Nat), Membership.mem H (Equiv.swap x ((HPow.hPow σ n) x))
    step3 : ∀ (y : α), Membership.mem H (Equiv.swap x y)
    step4 : ∀ (y z : α), Membership.mem H (Equiv.swap y z)
    τ : Equiv.Perm α
    y z : α
    left✝ : Ne y z
    h6 : Eq τ (Equiv.swap y z)
    ⊢ Membership.mem (↑(Subgroup.closure (Insert.insert σ (Singleton.singleton (Eq …
  -/
  exact step4 y z
  /-
    🎉 no goals
  -/


theorem closure_cycle_coprime_swap {n : ℕ} {σ : Perm α} (h0 : Nat.Coprime n (Fintype.card α))
    (h1 : IsCycle σ) (h2 : σ.support = Finset.univ) (x : α) :
    closure ({σ, swap x ((σ ^ n) x)} : Set (Perm α)) = ⊤ := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    σ : Equiv.Perm α
    h0 : n.Coprime (Fintype.card α)
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x ((H …
  -/
  rw [← Finset.card_univ, ← h2, ← h1.orderOf] at h0
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    σ : Equiv.Perm α
    h0 : n.Coprime (orderOf σ)
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x ((H …
  -/
  cases' exists_pow_eq_self_of_coprime h0 with m hm
  /-
    case intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    σ : Equiv.Perm α
    h0 : n.Coprime (orderOf σ)
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    m : Nat
    hm : Eq (HPow.hPow (HPow.hPow σ n) m) σ
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x ((H …
  -/
  have h2' : (σ ^ n).support = univ := Eq.trans (support_pow_coprime h0) h2
  /-
    case intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    σ : Equiv.Perm α
    h0 : n.Coprime (orderOf σ)
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    m : Nat
    hm : Eq (HPow.hPow (HPow.hPow σ n) m) σ
    h2' : Eq (HPow.hPow σ n).support Finset.univ
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x ((H …
  -/
  have h1' : IsCycle ((σ ^ n) ^ (m : ℤ)) := by rwa [← hm] at h1
  replace h1' : IsCycle (σ ^ n) :=
    h1'.of_pow (le_trans (support_pow_le σ n) (ge_of_eq (congr_arg support hm)))
  /-
    case intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    n : Nat
    σ : Equiv.Perm α
    h0 : n.Coprime (orderOf σ)
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x : α
    m : Nat
    hm : Eq (HPow.hPow (HPow.hPow σ n) m) σ
    h2' : Eq (HPow.hPow σ n).support Finset.univ
    h1' : (HPow.hPow σ n).IsCycle
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton (Equiv.swap x ((H …
  -/
  rw [eq_top_iff, ← closure_cycle_adjacent_swap h1' h2' x, closure_le, Set.insert_subset_iff]
  exact
    ⟨Subgroup.pow_mem (closure _) (subset_closure (Set.mem_insert σ _)) n,
      Set.singleton_subset_iff.mpr (subset_closure (Set.mem_insert_of_mem _ (Set.mem_singleton _)))⟩


theorem closure_prime_cycle_swap {σ τ : Perm α} (h0 : (Fintype.card α).Prime) (h1 : IsCycle σ)
    (h2 : σ.support = Finset.univ) (h3 : IsSwap τ) : closure ({σ, τ} : Set (Perm α)) = ⊤ := by
  /-
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    h3 : τ.IsSwap
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton τ))) Top.top
  -/
  obtain ⟨x, y, h4, h5⟩ := h3
  obtain ⟨i, hi⟩ :=
    h1.exists_pow_eq (mem_support.mp ((Finset.ext_iff.mp h2 x).mpr (Finset.mem_univ x)))
      (mem_support.mp ((Finset.ext_iff.mp h2 y).mpr (Finset.mem_univ y)))
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x y : α
    h4 : Ne x y
    h5 : Eq τ (Equiv.swap x y)
    i : Nat
    hi : Eq ((HPow.hPow σ i) x) y
    ⊢ Eq (Subgroup.closure (Insert.insert σ (Singleton.singleton τ))) Top.top
  -/
  rw [h5, ← hi]
  refine closure_cycle_coprime_swap
    (Nat.Coprime.symm (h0.coprime_iff_not_dvd.mpr fun h => h4 ?_)) h1 h2 x
  /-
    case intro.intro.intro.intro
    α : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    σ τ : Equiv.Perm α
    h0 : Nat.Prime (Fintype.card α)
    h1 : σ.IsCycle
    h2 : Eq σ.support Finset.univ
    x y : α
    h4 : Ne x y
    h5 : Eq τ (Equiv.swap x y)
    i : Nat
    hi : Eq ((HPow.hPow σ i) x) y
    h : Dvd.dvd (Fintype.card α) i
    ⊢ Eq x y
  -/
  cases' h with m hm
  rwa [hm, pow_mul, ← Finset.card_univ, ← h2, ← h1.orderOf, pow_orderOf_eq_one, one_pow,
    one_apply] at hi


