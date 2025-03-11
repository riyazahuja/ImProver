/-- Circulant graph over additive group `G` with jumps `s` -/
@[simps!]
def circulantGraph {G : Type*} [AddGroup G] (s : Set G) : SimpleGraph G :=
  fromRel (· - · ∈ s)


theorem circulantGraph_eq_erase_zero : circulantGraph s = circulantGraph (s \ {0}) := by
  /-
    G : Type u_1
    inst✝ : AddGroup G
    s : Set G
    ⊢ Eq (SimpleGraph.circulantGraph s) (SimpleGraph.circulantGraph (SDiff.sdiff s …
  -/
  ext (u v : G)
  /-
    case Adj.h.h.a
    G : Type u_1
    inst✝ : AddGroup G
    s : Set G
    u v : G
    ⊢ Iff ((SimpleGraph.circulantGraph s).Adj u v) ((SimpleGraph.circulantGraph (S …
  -/
  simp only [circulantGraph, fromRel_adj, and_congr_right_iff]
  /-
    case Adj.h.h.a
    G : Type u_1
    inst✝ : AddGroup G
    s : Set G
    u v : G
    ⊢ Ne u v → Iff (Or (Membership.mem s (HSub.hSub u v)) (Membership.mem s (HSub. …
  -/
  intro (h : u ≠ v)
  /-
    case Adj.h.h.a
    G : Type u_1
    inst✝ : AddGroup G
    s : Set G
    u v : G
    h : Ne u v
    ⊢ Iff (Or (Membership.mem s (HSub.hSub u v)) (Membership.mem s (HSub.hSub v u) …
  -/
  apply Iff.intro
    /-
      case Adj.h.h.a.mp
      G : Type u_1
      inst✝ : AddGroup G
      s : Set G
      u v : G
      h : Ne u v
      ⊢ Or (Membership.mem s (HSub.hSub u v)) (Membership.mem s (HSub.hSub v u)) → O …
    -/
  · intro h1
    cases h1 with
      | inl h1 => exact Or.inl ⟨h1, sub_ne_zero_of_ne h⟩
      | inr h1 => exact Or.inr ⟨h1, sub_ne_zero_of_ne h.symm⟩
    /-
      case Adj.h.h.a.mpr
      G : Type u_1
      inst✝ : AddGroup G
      s : Set G
      u v : G
      h : Ne u v
      ⊢ Or (Membership.mem (SDiff.sdiff s (Singleton.singleton 0)) (HSub.hSub u v))  …
    -/
  · intro h1
    cases h1 with
      | inl h1 => exact Or.inl h1.left
      | inr h1 => exact Or.inr h1.left


theorem circulantGraph_eq_symm : circulantGraph s = circulantGraph (s ∪ (-s)) := by
  /-
    G : Type u_1
    inst✝ : AddGroup G
    s : Set G
    ⊢ Eq (SimpleGraph.circulantGraph s) (SimpleGraph.circulantGraph (Union.union s …
  -/
  ext (u v : G)
  simp only [circulantGraph, fromRel_adj, Set.mem_union, Set.mem_neg, neg_sub, and_congr_right_iff,
    iff_self_or]
  /-
    case Adj.h.h.a
    G : Type u_1
    inst✝ : AddGroup G
    s : Set G
    u v : G
    ⊢ Ne u v → Or (Membership.mem s (HSub.hSub v u)) (Membership.mem s (HSub.hSub  …
  -/
  intro _ h
  /-
    case Adj.h.h.a
    G : Type u_1
    inst✝ : AddGroup G
    s : Set G
    u v : G
    a✝ : Ne u v
    h : Or (Membership.mem s (HSub.hSub v u)) (Membership.mem s (HSub.hSub u v))
    ⊢ Or (Membership.mem s (HSub.hSub u v)) (Membership.mem s (HSub.hSub v u))
  -/
  exact Or.symm h
  /-
    🎉 no goals
  -/


instance [DecidableEq G] [DecidablePred (· ∈ s)] : DecidableRel (circulantGraph s).Adj :=
  fun _ _ => inferInstanceAs (Decidable (_ ∧ _))


theorem circulantGraph_adj_translate {s : Set G} {u v d : G} :
                                                                              /-
                                                                                G : Type u_1
                                                                                inst✝ : AddGroup G
                                                                                s : Set G
                                                                                u v d : G
                                                                                ⊢ Iff ((SimpleGraph.circulantGraph s).Adj (HAdd.hAdd u d) (HAdd.hAdd v d)) ((S …
                                                                              -/
    (circulantGraph s).Adj (u + d) (v + d) ↔ (circulantGraph s).Adj u v := by simp
                                                                              /-
                                                                                🎉 no goals
                                                                              -/


/-- Cycle graph over `Fin n` -/
def cycleGraph : (n : ℕ) → SimpleGraph (Fin n)
  | 0 => ⊥
  | _ + 1 => circulantGraph {1}


instance : (n : ℕ) → DecidableRel (cycleGraph n).Adj
  | 0 => fun _ _ => inferInstanceAs (Decidable False)
  | _ + 1 => inferInstanceAs (DecidableRel (circulantGraph _).Adj)


theorem cycleGraph_zero_adj {u v : Fin 0} : ¬(cycleGraph 0).Adj u v := id


theorem cycleGraph_zero_eq_bot : cycleGraph 0 = ⊥ := Subsingleton.elim _ _

theorem cycleGraph_one_eq_bot : cycleGraph 1 = ⊥ := Subsingleton.elim _ _

theorem cycleGraph_zero_eq_top : cycleGraph 0 = ⊤ := Subsingleton.elim _ _

theorem cycleGraph_one_eq_top : cycleGraph 1 = ⊤ := Subsingleton.elim _ _


theorem cycleGraph_two_eq_top : cycleGraph 2 = ⊤ := by
  /-
    ⊢ Eq (SimpleGraph.cycleGraph 2) Top.top
  -/
  simp only [SimpleGraph.ext_iff, funext_iff]
  /-
    ⊢ ∀ (x x_1 : Fin 2), Eq ((SimpleGraph.cycleGraph 2).Adj x x_1) (Top.top.Adj x  …
  -/
  decide
  /-
    🎉 no goals
  -/


theorem cycleGraph_three_eq_top : cycleGraph 3 = ⊤ := by
  /-
    ⊢ Eq (SimpleGraph.cycleGraph 3) Top.top
  -/
  simp only [SimpleGraph.ext_iff, funext_iff]
  /-
    ⊢ ∀ (x x_1 : Fin 3), Eq ((SimpleGraph.cycleGraph 3).Adj x x_1) (Top.top.Adj x  …
  -/
  decide
  /-
    🎉 no goals
  -/


theorem cycleGraph_one_adj {u v : Fin 1} : ¬(cycleGraph 1).Adj u v := by
  /-
    u v : Fin 1
    ⊢ Not ((SimpleGraph.cycleGraph 1).Adj u v)
  -/
  rw [cycleGraph_one_eq_bot]
  /-
    u v : Fin 1
    ⊢ Not (Bot.bot.Adj u v)
  -/
  exact id
  /-
    🎉 no goals
  -/


theorem cycleGraph_adj {n : ℕ} {u v : Fin (n + 2)} :
    (cycleGraph (n + 2)).Adj u v ↔ u - v = 1 ∨ v - u = 1 := by
  /-
    n : Nat
    u v : Fin (HAdd.hAdd n 2)
    ⊢ Iff ((SimpleGraph.cycleGraph (HAdd.hAdd n 2)).Adj u v) (Or (Eq (HSub.hSub u  …
  -/
  simp only [cycleGraph, circulantGraph_adj, Set.mem_singleton_iff, and_iff_right_iff_imp]
  /-
    n : Nat
    u v : Fin (HAdd.hAdd n 2)
    ⊢ Or (Eq (HSub.hSub u v) 1) (Eq (HSub.hSub v u) 1) → Not (Eq u v)
  -/
  intro _ _
  /-
    n : Nat
    u v : Fin (HAdd.hAdd n 2)
    a✝¹ : Or (Eq (HSub.hSub u v) 1) (Eq (HSub.hSub v u) 1)
    a✝ : Eq u v
    ⊢ False
  -/
  simp_all
  /-
    🎉 no goals
  -/


theorem cycleGraph_adj' {n : ℕ} {u v : Fin n} :
    (cycleGraph n).Adj u v ↔ (u - v).val = 1 ∨ (v - u).val = 1 := by
  match n with
  | 0 => exact u.elim0
  | 1 => simp [cycleGraph_one_adj]
  | n + 2 => simp [cycleGraph_adj, Fin.ext_iff]


theorem cycleGraph_neighborSet {n : ℕ} {v : Fin (n + 2)} :
    (cycleGraph (n + 2)).neighborSet v = {v - 1, v + 1} := by
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 2)
    ⊢ Eq ((SimpleGraph.cycleGraph (HAdd.hAdd n 2)).neighborSet v) (Insert.insert ( …
  -/
  ext w
  /-
    case h
    n : Nat
    v w : Fin (HAdd.hAdd n 2)
    ⊢ Iff (Membership.mem ((SimpleGraph.cycleGraph (HAdd.hAdd n 2)).neighborSet v) …
  -/
  simp only [mem_neighborSet, Set.mem_insert_iff, Set.mem_singleton_iff]
  /-
    case h
    n : Nat
    v w : Fin (HAdd.hAdd n 2)
    ⊢ Iff ((SimpleGraph.cycleGraph (HAdd.hAdd n 2)).Adj v w) (Or (Eq w (HSub.hSub  …
  -/
  rw [cycleGraph_adj, sub_eq_iff_eq_add', sub_eq_iff_eq_add', eq_sub_iff_add_eq, eq_comm]
  /-
    🎉 no goals
  -/


theorem cycleGraph_neighborFinset {n : ℕ} {v : Fin (n + 2)} :
    (cycleGraph (n + 2)).neighborFinset v = {v - 1, v + 1} := by
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 2)
    ⊢ Eq ((SimpleGraph.cycleGraph (HAdd.hAdd n 2)).neighborFinset v) (Insert.inser …
  -/
  simp [neighborFinset, cycleGraph_neighborSet]
  /-
    🎉 no goals
  -/


theorem cycleGraph_degree_two_le {n : ℕ} {v : Fin (n + 2)} :
    (cycleGraph (n + 2)).degree v = Finset.card {v - 1, v + 1} := by
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 2)
    ⊢ Eq ((SimpleGraph.cycleGraph (HAdd.hAdd n 2)).degree v) (Insert.insert (HSub. …
  -/
  rw [SimpleGraph.degree, cycleGraph_neighborFinset]
  /-
    🎉 no goals
  -/


theorem cycleGraph_degree_three_le {n : ℕ} {v : Fin (n + 3)} :
    (cycleGraph (n + 3)).degree v = 2 := by
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 3)
    ⊢ Eq ((SimpleGraph.cycleGraph (HAdd.hAdd n 3)).degree v) 2
  -/
  rw [cycleGraph_degree_two_le, Finset.card_pair]
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 3)
    ⊢ Ne (HSub.hSub v 1) (HAdd.hAdd v 1)
  -/
  simp only [ne_eq, sub_eq_iff_eq_add, add_assoc v, self_eq_add_right]
  /-
    n : Nat
    v : Fin (HAdd.hAdd n 3)
    ⊢ Not (Eq (HAdd.hAdd 1 1) 0)
  -/
  exact ne_of_beq_false rfl
  /-
    🎉 no goals
  -/


theorem pathGraph_le_cycleGraph {n : ℕ} : pathGraph n ≤ cycleGraph n := by
  match n with
  | 0 | 1 => simp
  | n + 2 =>
    intro u v h
    rw [pathGraph_adj] at h
    rw [cycleGraph_adj']
    cases h with
    | inl h | inr h =>
      simp [Fin.coe_sub_iff_le.mpr (Nat.lt_of_succ_le h.le).le, Nat.eq_sub_of_add_eq' h]


theorem cycleGraph_preconnected {n : ℕ} : (cycleGraph n).Preconnected :=
  (pathGraph_preconnected n).mono pathGraph_le_cycleGraph


theorem cycleGraph_connected {n : ℕ} : (cycleGraph (n + 1)).Connected :=
  (pathGraph_connected n).mono pathGraph_le_cycleGraph


