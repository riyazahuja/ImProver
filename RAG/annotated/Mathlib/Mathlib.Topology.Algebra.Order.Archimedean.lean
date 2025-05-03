/-- Rational numbers are dense in a linear ordered archimedean field. -/
theorem Rat.denseRange_cast {𝕜} [LinearOrderedField 𝕜] [TopologicalSpace 𝕜] [OrderTopology 𝕜]
    [Archimedean 𝕜] : DenseRange ((↑) : ℚ → 𝕜) :=
  dense_of_exists_between fun _ _ h => Set.exists_range_iff.2 <| exists_rat_btwn h


/-- An additive subgroup of an archimedean linear ordered additive commutative group with order
topology is dense provided that for all positive `ε` there exists a positive element of the
subgroup that is less than `ε`. -/
theorem dense_of_not_isolated_zero (S : AddSubgroup G) (hS : ∀ ε > 0, ∃ g ∈ S, g ∈ Ioo 0 ε) :
    Dense (S : Set G) := by
  /-
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
    ⊢ Dense ↑S
  -/
  cases subsingleton_or_nontrivial G
    /-
      case inl
      G : Type u_1
      inst✝³ : LinearOrderedAddCommGroup G
      inst✝² : TopologicalSpace G
      inst✝¹ : OrderTopology G
      inst✝ : Archimedean G
      S : AddSubgroup G
      hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
      h✝ : Subsingleton G
      ⊢ Dense ↑S
    -/
  · refine fun x => _root_.subset_closure ?_
    /-
      case inl
      G : Type u_1
      inst✝³ : LinearOrderedAddCommGroup G
      inst✝² : TopologicalSpace G
      inst✝¹ : OrderTopology G
      inst✝ : Archimedean G
      S : AddSubgroup G
      hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
      h✝ : Subsingleton G
      x : G
      ⊢ Membership.mem (↑S) x
    -/
    rw [Subsingleton.elim x 0]
    /-
      case inl
      G : Type u_1
      inst✝³ : LinearOrderedAddCommGroup G
      inst✝² : TopologicalSpace G
      inst✝¹ : OrderTopology G
      inst✝ : Archimedean G
      S : AddSubgroup G
      hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
      h✝ : Subsingleton G
      x : G
      ⊢ Membership.mem (↑S) 0
    -/
    exact zero_mem S
    /-
      🎉 no goals
    -/
  /-
    case inr
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
    h✝ : Nontrivial G
    ⊢ Dense ↑S
  -/
  refine dense_of_exists_between fun a b hlt => ?_
  /-
    case inr
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
    h✝ : Nontrivial G
    a b : G
    hlt : LT.lt a b
    ⊢ Exists fun c => And (Membership.mem (↑S) c) (And (LT.lt a c) (LT.lt c b))
  -/
  rcases hS (b - a) (sub_pos.2 hlt) with ⟨g, hgS, hg0, hg⟩
  /-
    case inr.intro.intro.intro
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
    h✝ : Nontrivial G
    a b : G
    hlt : LT.lt a b
    g : G
    hgS : Membership.mem S g
    hg0 : LT.lt 0 g
    hg : LT.lt g (HSub.hSub b a)
    ⊢ Exists fun c => And (Membership.mem (↑S) c) (And (LT.lt a c) (LT.lt c b))
  -/
  rcases (existsUnique_add_zsmul_mem_Ioc hg0 0 a).exists with ⟨m, hm⟩
  /-
    case inr.intro.intro.intro.intro
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
    h✝ : Nontrivial G
    a b : G
    hlt : LT.lt a b
    g : G
    hgS : Membership.mem S g
    hg0 : LT.lt 0 g
    hg : LT.lt g (HSub.hSub b a)
    m : Int
    hm : Membership.mem (Set.Ioc a (HAdd.hAdd a g)) (HAdd.hAdd 0 (HSMul.hSMul m g))
    ⊢ Exists fun c => And (Membership.mem (↑S) c) (And (LT.lt a c) (LT.lt c b))
  -/
  rw [zero_add] at hm
  /-
    case inr.intro.intro.intro.intro
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
    h✝ : Nontrivial G
    a b : G
    hlt : LT.lt a b
    g : G
    hgS : Membership.mem S g
    hg0 : LT.lt 0 g
    hg : LT.lt g (HSub.hSub b a)
    m : Int
    hm : Membership.mem (Set.Ioc a (HAdd.hAdd a g)) (HSMul.hSMul m g)
    ⊢ Exists fun c => And (Membership.mem (↑S) c) (And (LT.lt a c) (LT.lt c b))
  -/
  refine ⟨m • g, zsmul_mem hgS _, hm.1, hm.2.trans_lt ?_⟩
  /-
    case inr.intro.intro.intro.intro
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hS : ∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Membersh …
    h✝ : Nontrivial G
    a b : G
    hlt : LT.lt a b
    g : G
    hgS : Membership.mem S g
    hg0 : LT.lt 0 g
    hg : LT.lt g (HSub.hSub b a)
    m : Int
    hm : Membership.mem (Set.Ioc a (HAdd.hAdd a g)) (HSMul.hSMul m g)
    ⊢ LT.lt (HAdd.hAdd a g) b
  -/
  rwa [lt_sub_iff_add_lt'] at hg
  /-
    🎉 no goals
  -/


/-- Let `S` be a nontrivial additive subgroup in an archimedean linear ordered additive commutative
group `G` with order topology. If the set of positive elements of `S` does not have a minimal
element, then `S` is dense `G`. -/
theorem dense_of_no_min (S : AddSubgroup G) (hbot : S ≠ ⊥)
    (H : ¬∃ a : G, IsLeast { g : G | g ∈ S ∧ 0 < g } a) : Dense (S : Set G) := by
  /-
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hbot : Ne S Bot.bot
    H : Not (Exists fun a => IsLeast (setOf fun g => And (Membership.mem S g) (LT. …
    ⊢ Dense ↑S
  -/
  refine S.dense_of_not_isolated_zero fun ε ε0 => ?_
  /-
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hbot : Ne S Bot.bot
    H : Not (Exists fun a => IsLeast (setOf fun g => And (Membership.mem S g) (LT. …
    ε : G
    ε0 : GT.gt ε 0
    ⊢ Exists fun g => And (Membership.mem S g) (Membership.mem (Set.Ioo 0 ε) g)
  -/
  contrapose! H
  /-
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    hbot : Ne S Bot.bot
    ε : G
    ε0 : GT.gt ε 0
    H : ∀ (g : G), Membership.mem S g → Not (Membership.mem (Set.Ioo 0 ε) g)
    ⊢ Exists fun a => IsLeast (setOf fun g => And (Membership.mem S g) (LT.lt 0 g) …
  -/
  exact exists_isLeast_pos hbot ε0 (disjoint_left.2 H)
  /-
    🎉 no goals
  -/


/-- An additive subgroup of an archimedean linear ordered additive commutative group `G` with order
topology either is dense in `G` or is a cyclic subgroup. -/
theorem dense_or_cyclic (S : AddSubgroup G) : Dense (S : Set G) ∨ ∃ a : G, S = closure {a} := by
  /-
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    ⊢ Or (Dense ↑S) (Exists fun a => Eq S (AddSubgroup.closure (Singleton.singleto …
  -/
  refine (em _).imp (dense_of_not_isolated_zero S) fun h => ?_
  /-
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    h : Not (∀ (ε : G), GT.gt ε 0 → Exists fun g => And (Membership.mem S g) (Memb …
    ⊢ Exists fun a => Eq S (AddSubgroup.closure (Singleton.singleton a))
  -/
  push_neg at h
  /-
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    h : Exists fun ε => And (GT.gt ε 0) (∀ (g : G), Membership.mem S g → Not (Memb …
    ⊢ Exists fun a => Eq S (AddSubgroup.closure (Singleton.singleton a))
  -/
  rcases h with ⟨ε, ε0, hε⟩
  /-
    case intro.intro
    G : Type u_1
    inst✝³ : LinearOrderedAddCommGroup G
    inst✝² : TopologicalSpace G
    inst✝¹ : OrderTopology G
    inst✝ : Archimedean G
    S : AddSubgroup G
    ε : G
    ε0 : GT.gt ε 0
    hε : ∀ (g : G), Membership.mem S g → Not (Membership.mem (Set.Ioo 0 ε) g)
    ⊢ Exists fun a => Eq S (AddSubgroup.closure (Singleton.singleton a))
  -/
  exact cyclic_of_isolated_zero ε0 (disjoint_left.2 hε)
  /-
    🎉 no goals
  -/


