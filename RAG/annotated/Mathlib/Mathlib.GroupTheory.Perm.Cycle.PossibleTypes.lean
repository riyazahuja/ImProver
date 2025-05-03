/-- There are permutations with cycleType `m` if and only if
  its sum is at most `Fintype.card α` and its members are at least 2. -/
theorem Equiv.Perm.exists_with_cycleType_iff {m : Multiset ℕ} :
    (∃ g : Equiv.Perm α, g.cycleType = m) ↔
      (m.sum ≤ Fintype.card α ∧ ∀ a ∈ m, 2 ≤ a) := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    inst✝ : Fintype α
    m : Multiset Nat
    ⊢ Iff (Exists fun g => Eq g.cycleType m) (And (LE.le m.sum (Fintype.card α)) ( …
  -/
  constructor
  · -- empty case
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      m : Multiset Nat
      ⊢ (Exists fun g => Eq g.cycleType m) → And (LE.le m.sum (Fintype.card α)) (∀ ( …
    -/
    intro h
    /-
      case mp
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      m : Multiset Nat
      h : Exists fun g => Eq g.cycleType m
      ⊢ And (LE.le m.sum (Fintype.card α)) (∀ (a : Nat), Membership.mem m a → LE.le  …
    -/
    obtain ⟨g, hg⟩ := h
    /-
      case mp.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      m : Multiset Nat
      g : Equiv.Perm α
      hg : Eq g.cycleType m
      ⊢ And (LE.le m.sum (Fintype.card α)) (∀ (a : Nat), Membership.mem m a → LE.le  …
    -/
    constructor
      /-
        case mp.intro.left
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        g : Equiv.Perm α
        hg : Eq g.cycleType m
        ⊢ LE.le m.sum (Fintype.card α)
      -/
    · rw [← hg, Equiv.Perm.sum_cycleType]
      /-
        case mp.intro.left
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        g : Equiv.Perm α
        hg : Eq g.cycleType m
        ⊢ LE.le g.support.card (Fintype.card α)
      -/
      exact (Equiv.Perm.support g).card_le_univ
      /-
        🎉 no goals
      -/
      /-
        case mp.intro.right
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        g : Equiv.Perm α
        hg : Eq g.cycleType m
        ⊢ ∀ (a : Nat), Membership.mem m a → LE.le 2 a
      -/
    · intro a
      /-
        case mp.intro.right
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        g : Equiv.Perm α
        hg : Eq g.cycleType m
        a : Nat
        ⊢ Membership.mem m a → LE.le 2 a
      -/
      rw [← hg]
      /-
        case mp.intro.right
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        g : Equiv.Perm α
        hg : Eq g.cycleType m
        a : Nat
        ⊢ Membership.mem g.cycleType a → LE.le 2 a
      -/
      exact Equiv.Perm.two_le_of_mem_cycleType
      /-
        🎉 no goals
      -/
    /-
      case mpr
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      m : Multiset Nat
      ⊢ And (LE.le m.sum (Fintype.card α)) (∀ (a : Nat), Membership.mem m a → LE.le  …
    -/
  · rintro ⟨hc, h2c⟩
    have hc' : m.toList.sum ≤ Fintype.card α := by
      simp only [Multiset.sum_toList]
      exact hc
    /-
      case mpr.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      m : Multiset Nat
      hc : LE.le m.sum (Fintype.card α)
      h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
      hc' : LE.le m.toList.sum (Fintype.card α)
      ⊢ Exists fun g => Eq g.cycleType m
    -/
    obtain ⟨p, hp_length, hp_nodup, hp_disj⟩ := List.exists_pw_disjoint_with_card hc'
    /-
      case mpr.intro.intro.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      m : Multiset Nat
      hc : LE.le m.sum (Fintype.card α)
      h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
      hc' : LE.le m.toList.sum (Fintype.card α)
      p : List (List α)
      hp_length : Eq (List.map List.length p) m.toList
      hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
      hp_disj : List.Pairwise List.Disjoint p
      ⊢ Exists fun g => Eq g.cycleType m
    -/
    use List.prod (List.map (fun l => List.formPerm l) p)
    have hp2 : ∀ x ∈ p, 2 ≤ x.length := by
      intro x hx
      apply h2c x.length
      rw [← Multiset.mem_toList, ← hp_length, List.mem_map]
      exact ⟨x, hx, rfl⟩
    /-
      case h
      α : Type u_1
      inst✝¹ : DecidableEq α
      inst✝ : Fintype α
      m : Multiset Nat
      hc : LE.le m.sum (Fintype.card α)
      h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
      hc' : LE.le m.toList.sum (Fintype.card α)
      p : List (List α)
      hp_length : Eq (List.map List.length p) m.toList
      hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
      hp_disj : List.Pairwise List.Disjoint p
      hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
      ⊢ Eq (List.map (fun l => l.formPerm) p).prod.cycleType m
    -/
    rw [Equiv.Perm.cycleType_eq _ rfl]
    · -- lengths
      /-
        case h
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ Eq (↑(List.map (Function.comp Finset.card Equiv.Perm.support) (List.map (fun …
      -/
      rw [← Multiset.coe_toList m]
      /-
        case h
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ Eq ↑(List.map (Function.comp Finset.card Equiv.Perm.support) (List.map (fun  …
      -/
      apply congr_arg
      /-
        case h.h
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ Eq (List.map (Function.comp Finset.card Equiv.Perm.support) (List.map (fun l …
      -/
      rw [List.map_map]; rw [← hp_length]
      /-
        case h.h
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ Eq (List.map (Function.comp (Function.comp Finset.card Equiv.Perm.support) f …
      -/
      apply List.map_congr_left
      /-
        case h.h.h
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ ∀ (a : List α), Membership.mem p a → Eq (Function.comp (Function.comp Finset …
      -/
      intro x hx; simp only [Function.comp_apply]
      /-
        case h.h.h
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        x : List α
        hx : Membership.mem p x
        ⊢ Eq x.formPerm.support.card x.length
      -/
      rw [List.support_formPerm_of_nodup x (hp_nodup x hx)]
      ·-- length
        /-
          case h.h.h
          α : Type u_1
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          m : Multiset Nat
          hc : LE.le m.sum (Fintype.card α)
          h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
          hc' : LE.le m.toList.sum (Fintype.card α)
          p : List (List α)
          hp_length : Eq (List.map List.length p) m.toList
          hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
          hp_disj : List.Pairwise List.Disjoint p
          hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
          x : List α
          hx : Membership.mem p x
          ⊢ Eq x.toFinset.card x.length
        -/
        rw [List.toFinset_card_of_nodup (hp_nodup x hx)]
        /-
          🎉 no goals
        -/
      · -- length >= 1
        /-
          case h.h.h
          α : Type u_1
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          m : Multiset Nat
          hc : LE.le m.sum (Fintype.card α)
          h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
          hc' : LE.le m.toList.sum (Fintype.card α)
          p : List (List α)
          hp_length : Eq (List.map List.length p) m.toList
          hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
          hp_disj : List.Pairwise List.Disjoint p
          hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
          x : List α
          hx : Membership.mem p x
          ⊢ ∀ (x_1 : α), Ne x (List.cons x_1 List.nil)
        -/
        intro a h
        /-
          case h.h.h
          α : Type u_1
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          m : Multiset Nat
          hc : LE.le m.sum (Fintype.card α)
          h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
          hc' : LE.le m.toList.sum (Fintype.card α)
          p : List (List α)
          hp_length : Eq (List.map List.length p) m.toList
          hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
          hp_disj : List.Pairwise List.Disjoint p
          hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
          x : List α
          hx : Membership.mem p x
          a : α
          h : Eq x (List.cons a List.nil)
          ⊢ False
        -/
        apply Nat.not_succ_le_self 1
        /-
          case h.h.h
          α : Type u_1
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          m : Multiset Nat
          hc : LE.le m.sum (Fintype.card α)
          h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
          hc' : LE.le m.toList.sum (Fintype.card α)
          p : List (List α)
          hp_length : Eq (List.map List.length p) m.toList
          hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
          hp_disj : List.Pairwise List.Disjoint p
          hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
          x : List α
          hx : Membership.mem p x
          a : α
          h : Eq x (List.cons a List.nil)
          ⊢ LE.le (Nat.succ 1) 1
        -/
        conv_rhs => rw [← List.length_singleton a]; rw [← h]
        /-
          case h.h.h
          α : Type u_1
          inst✝¹ : DecidableEq α
          inst✝ : Fintype α
          m : Multiset Nat
          hc : LE.le m.sum (Fintype.card α)
          h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
          hc' : LE.le m.toList.sum (Fintype.card α)
          p : List (List α)
          hp_length : Eq (List.map List.length p) m.toList
          hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
          hp_disj : List.Pairwise List.Disjoint p
          hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
          x : List α
          hx : Membership.mem p x
          a : α
          h : Eq x (List.cons a List.nil)
          ⊢ LE.le (Nat.succ 1) x.length
        -/
        exact hp2 x hx
        /-
          🎉 no goals
        -/
    · -- cycles
      /-
        case h.h1
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ ∀ (σ : Equiv.Perm α), Membership.mem (List.map (fun l => l.formPerm) p) σ →  …
      -/
      intro g
      /-
        case h.h1
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        g : Equiv.Perm α
        ⊢ Membership.mem (List.map (fun l => l.formPerm) p) g → g.IsCycle
      -/
      rw [List.mem_map]
      /-
        case h.h1
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        g : Equiv.Perm α
        ⊢ (Exists fun a => And (Membership.mem p a) (Eq a.formPerm g)) → g.IsCycle
      -/
      rintro ⟨x, hx, rfl⟩
      /-
        case h.h1.intro.intro
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        x : List α
        hx : Membership.mem p x
        ⊢ x.formPerm.IsCycle
      -/
      have hx_nodup : x.Nodup := hp_nodup x hx
      /-
        case h.h1.intro.intro
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        x : List α
        hx : Membership.mem p x
        hx_nodup : x.Nodup
        ⊢ x.formPerm.IsCycle
      -/
      rw [← Cycle.formPerm_coe x hx_nodup]
      /-
        case h.h1.intro.intro
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        x : List α
        hx : Membership.mem p x
        hx_nodup : x.Nodup
        ⊢ ((↑x).formPerm hx_nodup).IsCycle
      -/
      apply Cycle.isCycle_formPerm
      /-
        case h.h1.intro.intro.hn
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        x : List α
        hx : Membership.mem p x
        hx_nodup : x.Nodup
        ⊢ (↑x).Nontrivial
      -/
      rw [Cycle.nontrivial_coe_nodup_iff hx_nodup]
      /-
        case h.h1.intro.intro.hn
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        x : List α
        hx : Membership.mem p x
        hx_nodup : x.Nodup
        ⊢ LE.le 2 x.length
      -/
      exact hp2 x hx
      /-
        🎉 no goals
      -/
    · -- disjoint
      /-
        case h.h2
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ List.Pairwise Equiv.Perm.Disjoint (List.map (fun l => l.formPerm) p)
      -/
      rw [List.pairwise_map]
      /-
        case h.h2
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ List.Pairwise (fun a b => a.formPerm.Disjoint b.formPerm) p
      -/
      apply List.Pairwise.imp_of_mem _ hp_disj
      /-
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        ⊢ ∀ {a b : List α}, Membership.mem p a → Membership.mem p b → a.Disjoint b → a …
      -/
      intro a b ha hb hab
      /-
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        a b : List α
        ha : Membership.mem p a
        hb : Membership.mem p b
        hab : a.Disjoint b
        ⊢ a.formPerm.Disjoint b.formPerm
      -/
      rw [List.formPerm_disjoint_iff (hp_nodup a ha) (hp_nodup b hb) (hp2 a ha) (hp2 b hb)]
      /-
        α : Type u_1
        inst✝¹ : DecidableEq α
        inst✝ : Fintype α
        m : Multiset Nat
        hc : LE.le m.sum (Fintype.card α)
        h2c : ∀ (a : Nat), Membership.mem m a → LE.le 2 a
        hc' : LE.le m.toList.sum (Fintype.card α)
        p : List (List α)
        hp_length : Eq (List.map List.length p) m.toList
        hp_nodup : ∀ (s : List α), Membership.mem p s → s.Nodup
        hp_disj : List.Pairwise List.Disjoint p
        hp2 : ∀ (x : List α), Membership.mem p x → LE.le 2 x.length
        a b : List α
        ha : Membership.mem p a
        hb : Membership.mem p b
        hab : a.Disjoint b
        ⊢ a.Disjoint b
      -/
      exact hab
      /-
        🎉 no goals
      -/

