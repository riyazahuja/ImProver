/-- A hamiltonian path is a walk `p` that visits every vertex exactly once. Note that while
this definition doesn't contain that `p` is a path, `p.isPath` gives that. -/
def IsHamiltonian (p : G.Walk a b) : Prop := ∀ a, p.support.count a = 1


lemma IsHamiltonian.map {H : SimpleGraph β} (f : G →g H) (hf : Bijective f) (hp : p.IsHamiltonian) :
    (p.map f).IsHamiltonian := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    G : SimpleGraph α
    a b : α
    p : G.Walk a b
    H : SimpleGraph β
    f : G.Hom H
    hf : Function.Bijective ⇑f
    hp : p.IsHamiltonian
    ⊢ (SimpleGraph.Walk.map f p).IsHamiltonian
  -/
  simp [IsHamiltonian, hf.surjective.forall, hf.injective, hp _]
  /-
    🎉 no goals
  -/


/-- A hamiltonian path visits every vertex. -/
@[simp] lemma IsHamiltonian.mem_support (hp : p.IsHamiltonian) (c : α) : c ∈ p.support := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    G : SimpleGraph α
    a b : α
    p : G.Walk a b
    hp : p.IsHamiltonian
    c : α
    ⊢ Membership.mem p.support c
  -/
  simp only [← List.count_pos_iff, hp _, Nat.zero_lt_one]
  /-
    🎉 no goals
  -/


/-- Hamiltonian paths are paths. -/
lemma IsHamiltonian.isPath (hp : p.IsHamiltonian) : p.IsPath :=
  IsPath.mk' <| List.nodup_iff_count_le_one.2 <| (le_of_eq <| hp ·)


/-- A path whose support contains every vertex is hamiltonian. -/
lemma IsPath.isHamiltonian_of_mem (hp : p.IsPath) (hp' : ∀ w, w ∈ p.support) :
    p.IsHamiltonian := fun _ ↦
  le_antisymm (List.nodup_iff_count_le_one.1 hp.support_nodup _) (List.count_pos_iff.2 (hp' _))


lemma IsPath.isHamiltonian_iff (hp : p.IsPath) : p.IsHamiltonian ↔ ∀ w, w ∈ p.support :=
  ⟨(·.mem_support), hp.isHamiltonian_of_mem⟩


/-- The support of a hamiltonian walk is the entire vertex set. -/
lemma IsHamiltonian.support_toFinset (hp : p.IsHamiltonian) : p.support.toFinset = Finset.univ := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    G : SimpleGraph α
    a b : α
    p : G.Walk a b
    inst✝ : Fintype α
    hp : p.IsHamiltonian
    ⊢ Eq p.support.toFinset Finset.univ
  -/
  simp [eq_univ_iff_forall, hp]
  /-
    🎉 no goals
  -/


/-- The length of a hamiltonian path is one less than the number of vertices of the graph. -/
lemma IsHamiltonian.length_eq (hp : p.IsHamiltonian) : p.length = Fintype.card α - 1 :=
  eq_tsub_of_add_eq <| by
    rw [← length_support, ← List.sum_toFinset_count_eq_length, Finset.sum_congr rfl fun _ _ ↦ hp _,
      ← card_eq_sum_ones, hp.support_toFinset, card_univ]


/-- A hamiltonian cycle is a cycle that visits every vertex once. -/
structure IsHamiltonianCycle (p : G.Walk a a) extends p.IsCycle : Prop where
  isHamiltonian_tail : p.tail.IsHamiltonian


lemma IsHamiltonianCycle.isCycle (hp : p.IsHamiltonianCycle) : p.IsCycle :=
  hp.toIsCycle


lemma IsHamiltonianCycle.map {H : SimpleGraph β} (f : G →g H) (hf : Bijective f)
    (hp : p.IsHamiltonianCycle) : (p.map f).IsHamiltonianCycle where
  toIsCycle := hp.isCycle.map hf.injective
  isHamiltonian_tail := by
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      G : SimpleGraph α
      a : α
      p : G.Walk a a
      H : SimpleGraph β
      f : G.Hom H
      hf : Function.Bijective ⇑f
      hp : p.IsHamiltonianCycle
      ⊢ (SimpleGraph.Walk.map f p).tail.IsHamiltonian
    -/
    simp only [IsHamiltonian, hf.surjective.forall]
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      G : SimpleGraph α
      a : α
      p : G.Walk a a
      H : SimpleGraph β
      f : G.Hom H
      hf : Function.Bijective ⇑f
      hp : p.IsHamiltonianCycle
      ⊢ ∀ (x : α), Eq (List.count (f x) (SimpleGraph.Walk.map f p).tail.support) 1
    -/
    intro x
    /-
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      G : SimpleGraph α
      a : α
      p : G.Walk a a
      H : SimpleGraph β
      f : G.Hom H
      hf : Function.Bijective ⇑f
      hp : p.IsHamiltonianCycle
      x : α
      ⊢ Eq (List.count (f x) (SimpleGraph.Walk.map f p).tail.support) 1
    -/
    rcases p with (_ | ⟨y, p⟩)
      /-
        case nil
        α : Type u_1
        β : Type u_2
        inst✝¹ : DecidableEq α
        inst✝ : DecidableEq β
        G : SimpleGraph α
        a : α
        H : SimpleGraph β
        f : G.Hom H
        hf : Function.Bijective ⇑f
        x : α
        hp : SimpleGraph.Walk.nil.IsHamiltonianCycle
        ⊢ Eq (List.count (f x) (SimpleGraph.Walk.map f SimpleGraph.Walk.nil).tail.supp …
      -/
    · cases hp.ne_nil rfl
      /-
        🎉 no goals
      -/
    /-
      case cons
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      G : SimpleGraph α
      a : α
      H : SimpleGraph β
      f : G.Hom H
      hf : Function.Bijective ⇑f
      x v✝ : α
      y : G.Adj a v✝
      p : G.Walk v✝ a
      hp : (SimpleGraph.Walk.cons y p).IsHamiltonianCycle
      ⊢ Eq (List.count (f x) (SimpleGraph.Walk.map f (SimpleGraph.Walk.cons y p)).ta …
    -/
    simp only [map_cons, getVert_cons_succ, tail_cons_eq, support_copy,support_map]
    /-
      case cons
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      G : SimpleGraph α
      a : α
      H : SimpleGraph β
      f : G.Hom H
      hf : Function.Bijective ⇑f
      x v✝ : α
      y : G.Adj a v✝
      p : G.Walk v✝ a
      hp : (SimpleGraph.Walk.cons y p).IsHamiltonianCycle
      ⊢ Eq (List.count (f x) (List.map (⇑f) p.support)) 1
    -/
    rw [List.count_map_of_injective _ _ hf.injective, ← support_copy, ← tail_cons_eq]
    /-
      case cons
      α : Type u_1
      β : Type u_2
      inst✝¹ : DecidableEq α
      inst✝ : DecidableEq β
      G : SimpleGraph α
      a : α
      H : SimpleGraph β
      f : G.Hom H
      hf : Function.Bijective ⇑f
      x v✝ : α
      y : G.Adj a v✝
      p : G.Walk v✝ a
      hp : (SimpleGraph.Walk.cons y p).IsHamiltonianCycle
      ⊢ Eq (List.count x (SimpleGraph.Walk.cons ?cons.h p).tail.support) 1
    -/
    exact hp.isHamiltonian_tail _
    /-
      🎉 no goals
    -/


lemma isHamiltonianCycle_isCycle_and_isHamiltonian_tail  :
    p.IsHamiltonianCycle ↔ p.IsCycle ∧ p.tail.IsHamiltonian :=
  ⟨fun ⟨h, h'⟩ ↦ ⟨h, h'⟩, fun ⟨h, h'⟩ ↦ ⟨h, h'⟩⟩


lemma isHamiltonianCycle_iff_isCycle_and_support_count_tail_eq_one :
    p.IsHamiltonianCycle ↔ p.IsCycle ∧ ∀ a, (support p).tail.count a = 1 := by
  simp +contextual [isHamiltonianCycle_isCycle_and_isHamiltonian_tail,
    IsHamiltonian, support_tail, IsCycle.not_nil, exists_prop]


/-- A hamiltonian cycle visits every vertex. -/
lemma IsHamiltonianCycle.mem_support (hp : p.IsHamiltonianCycle) (b : α) :
    b ∈ p.support :=
  List.mem_of_mem_tail <| support_tail p hp.1.not_nil ▸ hp.isHamiltonian_tail.mem_support _


/-- The length of a hamiltonian cycle is the number of vertices. -/
lemma IsHamiltonianCycle.length_eq [Fintype α] (hp : p.IsHamiltonianCycle) :
    p.length = Fintype.card α := by
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    G : SimpleGraph α
    a : α
    p : G.Walk a a
    inst✝ : Fintype α
    hp : p.IsHamiltonianCycle
    ⊢ Eq p.length (Fintype.card α)
  -/
  rw [← length_tail_add_one hp.not_nil, hp.isHamiltonian_tail.length_eq, Nat.sub_add_cancel]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    G : SimpleGraph α
    a : α
    p : G.Walk a a
    inst✝ : Fintype α
    hp : p.IsHamiltonianCycle
    ⊢ LE.le 1 (Fintype.card α)
  -/
  rw [Nat.succ_le, Fintype.card_pos_iff]
  /-
    α : Type u_1
    inst✝¹ : DecidableEq α
    G : SimpleGraph α
    a : α
    p : G.Walk a a
    inst✝ : Fintype α
    hp : p.IsHamiltonianCycle
    ⊢ Nonempty α
  -/
  exact ⟨a⟩
  /-
    🎉 no goals
  -/


lemma IsHamiltonianCycle.count_support_self (hp : p.IsHamiltonianCycle) :
    p.support.count a = 2 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    G : SimpleGraph α
    a : α
    p : G.Walk a a
    hp : p.IsHamiltonianCycle
    ⊢ Eq (List.count a p.support) 2
  -/
  rw [support_eq_cons, List.count_cons_self, ← support_tail _ hp.1.not_nil, hp.isHamiltonian_tail]
  /-
    🎉 no goals
  -/


lemma IsHamiltonianCycle.support_count_of_ne (hp : p.IsHamiltonianCycle) (h : a ≠ b) :
    p.support.count b = 1 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    G : SimpleGraph α
    a b : α
    p : G.Walk a a
    hp : p.IsHamiltonianCycle
    h : Ne a b
    ⊢ Eq (List.count b p.support) 1
  -/
  rw [← cons_support_tail p hp.1.not_nil, List.count_cons_of_ne h.symm, hp.isHamiltonian_tail]
  /-
    🎉 no goals
  -/


/-- A hamiltonian graph is a graph that contains a hamiltonian cycle.

By convention, the singleton graph is considered to be hamiltonian. -/
def IsHamiltonian (G : SimpleGraph α) : Prop :=
  Fintype.card α ≠ 1 → ∃ a, ∃ p : G.Walk a a, p.IsHamiltonianCycle


lemma IsHamiltonian.mono {H : SimpleGraph α} (hGH : G ≤ H) (hG : G.IsHamiltonian) :
    H.IsHamiltonian :=
  fun hα ↦ let ⟨_, p, hp⟩ := hG hα; ⟨_, p.map <| .ofLE hGH, hp.map _ bijective_id⟩


lemma IsHamiltonian.connected (hG : G.IsHamiltonian) : G.Connected where
  preconnected a b := by
    /-
      α : Type u_1
      inst✝¹ : DecidableEq α
      G : SimpleGraph α
      inst✝ : Fintype α
      hG : G.IsHamiltonian
      a b : α
      ⊢ G.Reachable a b
    -/
    obtain rfl | hab := eq_or_ne a b
      /-
        case inl
        α : Type u_1
        inst✝¹ : DecidableEq α
        G : SimpleGraph α
        inst✝ : Fintype α
        hG : G.IsHamiltonian
        a : α
        ⊢ G.Reachable a a
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      case inr
      α : Type u_1
      inst✝¹ : DecidableEq α
      G : SimpleGraph α
      inst✝ : Fintype α
      hG : G.IsHamiltonian
      a b : α
      hab : Ne a b
      ⊢ G.Reachable a b
    -/
    have : Nontrivial α := ⟨a, b, hab⟩
    /-
      case inr
      α : Type u_1
      inst✝¹ : DecidableEq α
      G : SimpleGraph α
      inst✝ : Fintype α
      hG : G.IsHamiltonian
      a b : α
      hab : Ne a b
      this : Nontrivial α
      ⊢ G.Reachable a b
    -/
    obtain ⟨_, p, hp⟩ := hG Fintype.one_lt_card.ne'
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      G : SimpleGraph α
      inst✝ : Fintype α
      hG : G.IsHamiltonian
      a b : α
      hab : Ne a b
      this : Nontrivial α
      w✝ : α
      p : G.Walk w✝ w✝
      hp : p.IsHamiltonianCycle
      ⊢ G.Reachable a b
    -/
    have a_mem := hp.mem_support a
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      G : SimpleGraph α
      inst✝ : Fintype α
      hG : G.IsHamiltonian
      a b : α
      hab : Ne a b
      this : Nontrivial α
      w✝ : α
      p : G.Walk w✝ w✝
      hp : p.IsHamiltonianCycle
      a_mem : Membership.mem p.support a
      ⊢ G.Reachable a b
    -/
    have b_mem := hp.mem_support b
    /-
      case inr.intro.intro
      α : Type u_1
      inst✝¹ : DecidableEq α
      G : SimpleGraph α
      inst✝ : Fintype α
      hG : G.IsHamiltonian
      a b : α
      hab : Ne a b
      this : Nontrivial α
      w✝ : α
      p : G.Walk w✝ w✝
      hp : p.IsHamiltonianCycle
      a_mem : Membership.mem p.support a
      b_mem : Membership.mem p.support b
      ⊢ G.Reachable a b
    -/
    exact ((p.takeUntil a a_mem).reverse.append <| p.takeUntil b b_mem).reachable
    /-
      🎉 no goals
    -/
                                           /-
                                             α : Type u_1
                                             inst✝¹ : DecidableEq α
                                             G : SimpleGraph α
                                             inst✝ : Fintype α
                                             hG : G.IsHamiltonian
                                             x✝ : IsEmpty α
                                             ⊢ False
                                           -/
  nonempty := not_isEmpty_iff.1 fun _ ↦ by simpa using hG <| by simp [@Fintype.card_eq_zero]
                                           /-
                                             🎉 no goals
                                           -/


