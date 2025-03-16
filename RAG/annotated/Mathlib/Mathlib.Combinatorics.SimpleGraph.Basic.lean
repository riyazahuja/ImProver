/--
A variant of the `aesop` tactic for use in the graph library. Changes relative
to standard `aesop`:

- We use the `SimpleGraph` rule set in addition to the default rule sets.
- We instruct Aesop's `intro` rule to unfold with `default` transparency.
- We instruct Aesop to fail if it can't fully solve the goal. This allows us to
  use `aesop_graph` for auto-params.
-/
macro (name := aesop_graph) "aesop_graph" c:Aesop.tactic_clause* : tactic =>
  `(tactic|
    aesop $c*
      (config := { introsTransparency? := some .default, terminal := true })
      (rule_sets := [$(Lean.mkIdent `SimpleGraph):ident]))


/--
Use `aesop_graph?` to pass along a `Try this` suggestion when using `aesop_graph`
-/
macro (name := aesop_graph?) "aesop_graph?" c:Aesop.tactic_clause* : tactic =>
  `(tactic|
    aesop? $c*
      (config := { introsTransparency? := some .default, terminal := true })
      (rule_sets := [$(Lean.mkIdent `SimpleGraph):ident]))


/--
A variant of `aesop_graph` which does not fail if it is unable to solve the goal.
Use this only for exploration! Nonterminal Aesop is even worse than nonterminal `simp`.
-/
macro (name := aesop_graph_nonterminal) "aesop_graph_nonterminal" c:Aesop.tactic_clause* : tactic =>
  `(tactic|
    aesop $c*
      (config := { introsTransparency? := some .default, warnOnNonterminal := false })
      (rule_sets := [$(Lean.mkIdent `SimpleGraph):ident]))


/-- A simple graph is an irreflexive symmetric relation `Adj` on a vertex type `V`.
The relation describes which pairs of vertices are adjacent.
There is exactly one edge for every pair of adjacent vertices;
see `SimpleGraph.edgeSet` for the corresponding edge set.
-/
@[ext, aesop safe constructors (rule_sets := [SimpleGraph])]
structure SimpleGraph (V : Type u) where
  /-- The adjacency relation of a simple graph. -/
  Adj : V → V → Prop
  symm : Symmetric Adj := by aesop_graph
  loopless : Irreflexive Adj := by aesop_graph
-- Porting note: changed `obviously` to `aesop` in the `structure`


/-- Constructor for simple graphs using a symmetric irreflexive boolean function. -/
@[simps]
def SimpleGraph.mk' {V : Type u} :
    {adj : V → V → Bool // (∀ x y, adj x y = adj y x) ∧ (∀ x, ¬ adj x x)} ↪ SimpleGraph V where
                                              /-
                                                V : Type u
                                                x : Subtype fun adj => And (∀ (x y : V), Eq (adj x y) (adj y x)) (∀ (x : V), N …
                                                v w : V
                                                ⊢ (fun v w => Eq (↑x v w) Bool.true) v w → (fun v w => Eq (↑x v w) Bool.true)  …
                                              -/
                                              /-
                                                🎉 no goals
                                              -/
  toFun x := ⟨fun v w ↦ x.1 v w, fun v w ↦ by simp [x.2.1], fun v ↦ by simp [x.2.2]⟩
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  inj' := by
    /-
      V : Type u
      ⊢ Function.Injective fun x => { Adj := fun v w => Eq (↑x v w) Bool.true, symm  …
    -/
    rintro ⟨adj, _⟩ ⟨adj', _⟩
    /-
      case mk.mk
      V : Type u
      adj : V → V → Bool
      property✝¹ : And (∀ (x y : V), Eq (adj x y) (adj y x)) (∀ (x : V), Not (Eq (ad …
      adj' : V → V → Bool
      property✝ : And (∀ (x y : V), Eq (adj' x y) (adj' y x)) (∀ (x : V), Not (Eq (a …
      ⊢ Eq ((fun x => { Adj := fun v w => Eq (↑x v w) Bool.true, symm := ⋯, loopless …
    -/
    simp only [mk.injEq, Subtype.mk.injEq]
    /-
      case mk.mk
      V : Type u
      adj : V → V → Bool
      property✝¹ : And (∀ (x y : V), Eq (adj x y) (adj y x)) (∀ (x : V), Not (Eq (ad …
      adj' : V → V → Bool
      property✝ : And (∀ (x y : V), Eq (adj' x y) (adj' y x)) (∀ (x : V), Not (Eq (a …
      ⊢ (Eq (fun v w => Eq (adj v w) Bool.true) fun v w => Eq (adj' v w) Bool.true)  …
    -/
    intro h
    /-
      case mk.mk
      V : Type u
      adj : V → V → Bool
      property✝¹ : And (∀ (x y : V), Eq (adj x y) (adj y x)) (∀ (x : V), Not (Eq (ad …
      adj' : V → V → Bool
      property✝ : And (∀ (x y : V), Eq (adj' x y) (adj' y x)) (∀ (x : V), Not (Eq (a …
      h : Eq (fun v w => Eq (adj v w) Bool.true) fun v w => Eq (adj' v w) Bool.true
      ⊢ Eq adj adj'
    -/
    funext v w
    /-
      case mk.mk.h.h
      V : Type u
      adj : V → V → Bool
      property✝¹ : And (∀ (x y : V), Eq (adj x y) (adj y x)) (∀ (x : V), Not (Eq (ad …
      adj' : V → V → Bool
      property✝ : And (∀ (x y : V), Eq (adj' x y) (adj' y x)) (∀ (x : V), Not (Eq (a …
      h : Eq (fun v w => Eq (adj v w) Bool.true) fun v w => Eq (adj' v w) Bool.true
      v w : V
      ⊢ Eq (adj v w) (adj' v w)
    -/
    simpa [Bool.coe_iff_coe] using congr_fun₂ h v w
    /-
      🎉 no goals
    -/


/-- We can enumerate simple graphs by enumerating all functions `V → V → Bool`
and filtering on whether they are symmetric and irreflexive. -/
instance {V : Type u} [Fintype V] [DecidableEq V] : Fintype (SimpleGraph V) where
  elems := Finset.univ.map SimpleGraph.mk'
  complete := by
    classical
    rintro ⟨Adj, hs, hi⟩
    simp only [mem_map, mem_univ, true_and, Subtype.exists, Bool.not_eq_true]
    refine ⟨fun v w ↦ Adj v w, ⟨?_, ?_⟩, ?_⟩
    · simp [hs.iff]
    · intro v; simp [hi v]
    · ext
      simp


/-- There are finitely many simple graphs on a given finite type. -/
instance SimpleGraph.instFinite {V : Type u} [Finite V] : Finite (SimpleGraph V) :=
  .of_injective SimpleGraph.Adj fun _ _ ↦ SimpleGraph.ext


/-- Construct the simple graph induced by the given relation. It
symmetrizes the relation and makes it irreflexive. -/
def SimpleGraph.fromRel {V : Type u} (r : V → V → Prop) : SimpleGraph V where
  Adj a b := a ≠ b ∧ (r a b ∨ r b a)
  symm := fun _ _ ⟨hn, hr⟩ => ⟨hn.symm, hr.symm⟩
  loopless := fun _ ⟨hn, _⟩ => hn rfl


@[simp]
theorem SimpleGraph.fromRel_adj {V : Type u} (r : V → V → Prop) (v w : V) :
    (SimpleGraph.fromRel r).Adj v w ↔ v ≠ w ∧ (r v w ∨ r w v) :=
  Iff.rfl

-- Porting note: attributes needed for `completeGraph`

/-- The complete graph on a type `V` is the simple graph with all pairs of distinct vertices
adjacent. In `Mathlib`, this is usually referred to as `⊤`. -/
def completeGraph (V : Type u) : SimpleGraph V where Adj := Ne


/-- The graph with no edges on a given vertex type `V`. `Mathlib` prefers the notation `⊥`. -/
def emptyGraph (V : Type u) : SimpleGraph V where Adj _ _ := False


/-- Two vertices are adjacent in the complete bipartite graph on two vertex types
if and only if they are not from the same side.
Any bipartite graph may be regarded as a subgraph of one of these. -/
@[simps]
def completeBipartiteGraph (V W : Type*) : SimpleGraph (V ⊕ W) where
  Adj v w := v.isLeft ∧ w.isRight ∨ v.isRight ∧ w.isLeft
                 /-
                   V : Type u_1
                   W : Type u_2
                   v w : Sum V W
                   ⊢ (fun v w => Or (And (Eq v.isLeft Bool.true) (Eq w.isRight Bool.true)) (And ( …
                 -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
                                         /-
                                           🎉 no goals
                                         -/
  symm v w := by cases v <;> cases w <;> simp
                                         /-
                                           🎉 no goals
                                         -/
                   /-
                     V : Type u_1
                     W : Type u_2
                     v : Sum V W
                     ⊢ Not ((fun v w => Or (And (Eq v.isLeft Bool.true) (Eq w.isRight Bool.true)) ( …
                   -/
                               /-
                                 🎉 no goals
                               -/
  loopless v := by cases v <;> simp
                               /-
                                 🎉 no goals
                               -/


@[simp]
protected theorem irrefl {v : V} : ¬G.Adj v v :=
  G.loopless v


theorem adj_comm (u v : V) : G.Adj u v ↔ G.Adj v u :=
  ⟨fun x => G.symm x, fun x => G.symm x⟩


@[symm]
theorem adj_symm (h : G.Adj u v) : G.Adj v u :=
  G.symm h


theorem Adj.symm {G : SimpleGraph V} {u v : V} (h : G.Adj u v) : G.Adj v u :=
  G.symm h


theorem ne_of_adj (h : G.Adj a b) : a ≠ b := by
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    h : G.Adj a b
    ⊢ Ne a b
  -/
  rintro rfl
  /-
    V : Type u
    G : SimpleGraph V
    a : V
    h : G.Adj a a
    ⊢ False
  -/
  exact G.irrefl h
  /-
    🎉 no goals
  -/


protected theorem Adj.ne {G : SimpleGraph V} {a b : V} (h : G.Adj a b) : a ≠ b :=
  G.ne_of_adj h


protected theorem Adj.ne' {G : SimpleGraph V} {a b : V} (h : G.Adj a b) : b ≠ a :=
  h.ne.symm


theorem ne_of_adj_of_not_adj {v w x : V} (h : G.Adj v x) (hn : ¬G.Adj w x) : v ≠ w := fun h' =>
  hn (h' ▸ h)


theorem adj_injective : Injective (Adj : SimpleGraph V → V → V → Prop) :=
  fun _ _ => SimpleGraph.ext


@[simp]
theorem adj_inj {G H : SimpleGraph V} : G.Adj = H.Adj ↔ G = H :=
  adj_injective.eq_iff


/-- The relation that one `SimpleGraph` is a subgraph of another.
Note that this should be spelled `≤`. -/
def IsSubgraph (x y : SimpleGraph V) : Prop :=
  ∀ ⦃v w : V⦄, x.Adj v w → y.Adj v w


instance : LE (SimpleGraph V) :=
  ⟨IsSubgraph⟩


@[simp]
theorem isSubgraph_eq_le : (IsSubgraph : SimpleGraph V → SimpleGraph V → Prop) = (· ≤ ·) :=
  rfl


/-- The supremum of two graphs `x ⊔ y` has edges where either `x` or `y` have edges. -/
instance : Max (SimpleGraph V) where
  max x y :=
    { Adj := x.Adj ⊔ y.Adj
                              /-
                                ι : Sort u_1
                                V : Type u
                                G : SimpleGraph V
                                a b c u v✝ w✝ : V
                                e : Sym2 V
                                x y : SimpleGraph V
                                v w : V
                                h : Max.max x.Adj y.Adj v w
                                ⊢ Max.max x.Adj y.Adj w v
                              -/
      symm := fun v w h => by rwa [Pi.sup_apply, Pi.sup_apply, x.adj_comm, y.adj_comm] }
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem sup_adj (x y : SimpleGraph V) (v w : V) : (x ⊔ y).Adj v w ↔ x.Adj v w ∨ y.Adj v w :=
  Iff.rfl


/-- The infimum of two graphs `x ⊓ y` has edges where both `x` and `y` have edges. -/
instance : Min (SimpleGraph V) where
  min x y :=
    { Adj := x.Adj ⊓ y.Adj
                              /-
                                ι : Sort u_1
                                V : Type u
                                G : SimpleGraph V
                                a b c u v✝ w✝ : V
                                e : Sym2 V
                                x y : SimpleGraph V
                                v w : V
                                h : Min.min x.Adj y.Adj v w
                                ⊢ Min.min x.Adj y.Adj w v
                              -/
      symm := fun v w h => by rwa [Pi.inf_apply, Pi.inf_apply, x.adj_comm, y.adj_comm] }
                              /-
                                🎉 no goals
                              -/


@[simp]
theorem inf_adj (x y : SimpleGraph V) (v w : V) : (x ⊓ y).Adj v w ↔ x.Adj v w ∧ y.Adj v w :=
  Iff.rfl


/-- We define `Gᶜ` to be the `SimpleGraph V` such that no two adjacent vertices in `G`
are adjacent in the complement, and every nonadjacent pair of vertices is adjacent
(still ensuring that vertices are not adjacent to themselves). -/
instance hasCompl : HasCompl (SimpleGraph V) where
  compl G :=
    { Adj := fun v w => v ≠ w ∧ ¬G.Adj v w
                                                /-
                                                  ι : Sort u_1
                                                  V : Type u
                                                  G✝ : SimpleGraph V
                                                  a b c u v✝ w✝ : V
                                                  e : Sym2 V
                                                  G : SimpleGraph V
                                                  v w : V
                                                  x✝ : (fun v w => And (Ne v w) (Not (G.Adj v w))) v w
                                                  hne : Ne v w
                                                  right✝ : Not (G.Adj v w)
                                                  ⊢ Not (G.Adj w v)
                                                -/
      symm := fun v w ⟨hne, _⟩ => ⟨hne.symm, by rwa [adj_comm]⟩
                                                /-
                                                  🎉 no goals
                                                -/
      loopless := fun _ ⟨hne, _⟩ => (hne rfl).elim }


@[simp]
theorem compl_adj (G : SimpleGraph V) (v w : V) : Gᶜ.Adj v w ↔ v ≠ w ∧ ¬G.Adj v w :=
  Iff.rfl


/-- The difference of two graphs `x \ y` has the edges of `x` with the edges of `y` removed. -/
instance sdiff : SDiff (SimpleGraph V) where
  sdiff x y :=
    { Adj := x.Adj \ y.Adj
                              /-
                                ι : Sort u_1
                                V : Type u
                                G : SimpleGraph V
                                a b c u v✝ w✝ : V
                                e : Sym2 V
                                x y : SimpleGraph V
                                v w : V
                                h : SDiff.sdiff x.Adj y.Adj v w
                                ⊢ SDiff.sdiff x.Adj y.Adj w v
                              -/
      symm := fun v w h => by change x.Adj w v ∧ ¬y.Adj w v; rwa [x.adj_comm, y.adj_comm] }
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
theorem sdiff_adj (x y : SimpleGraph V) (v w : V) : (x \ y).Adj v w ↔ x.Adj v w ∧ ¬y.Adj v w :=
  Iff.rfl


instance supSet : SupSet (SimpleGraph V) where
  sSup s :=
    { Adj := fun a b => ∃ G ∈ s, Adj G a b
      symm := fun _ _ => Exists.imp fun _ => And.imp_right Adj.symm
      loopless := by
        /-
          ι : Sort u_1
          V : Type u
          G : SimpleGraph V
          a b c u v w : V
          e : Sym2 V
          s : Set (SimpleGraph V)
          ⊢ Irreflexive fun a b => Exists fun G => And (Membership.mem s G) (G.Adj a b)
        -/
        rintro a ⟨G, _, ha⟩
        /-
          case intro.intro
          ι : Sort u_1
          V : Type u
          G✝ : SimpleGraph V
          a✝ b c u v w : V
          e : Sym2 V
          s : Set (SimpleGraph V)
          a : V
          G : SimpleGraph V
          left✝ : Membership.mem s G
          ha : G.Adj a a
          ⊢ False
        -/
        exact ha.ne rfl }
        /-
          🎉 no goals
        -/


instance infSet : InfSet (SimpleGraph V) where
  sInf s :=
    { Adj := fun a b => (∀ ⦃G⦄, G ∈ s → Adj G a b) ∧ a ≠ b
      symm := fun _ _ => And.imp (forall₂_imp fun _ _ => Adj.symm) Ne.symm
      loopless := fun _ h => h.2 rfl }


@[simp]
theorem sSup_adj {s : Set (SimpleGraph V)} {a b : V} : (sSup s).Adj a b ↔ ∃ G ∈ s, Adj G a b :=
  Iff.rfl


@[simp]
theorem sInf_adj {s : Set (SimpleGraph V)} : (sInf s).Adj a b ↔ (∀ G ∈ s, Adj G a b) ∧ a ≠ b :=
  Iff.rfl


@[simp]
                                                                                         /-
                                                                                           ι : Sort u_1
                                                                                           V : Type u
                                                                                           a b : V
                                                                                           f : ι → SimpleGraph V
                                                                                           ⊢ Iff ((iSup fun i => f i).Adj a b) (Exists fun i => (f i).Adj a b)
                                                                                         -/
theorem iSup_adj {f : ι → SimpleGraph V} : (⨆ i, f i).Adj a b ↔ ∃ i, (f i).Adj a b := by simp [iSup]
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/


@[simp]
theorem iInf_adj {f : ι → SimpleGraph V} : (⨅ i, f i).Adj a b ↔ (∀ i, (f i).Adj a b) ∧ a ≠ b := by
  /-
    ι : Sort u_1
    V : Type u
    a b : V
    f : ι → SimpleGraph V
    ⊢ Iff ((iInf fun i => f i).Adj a b) (And (∀ (i : ι), (f i).Adj a b) (Ne a b))
  -/
  simp [iInf]
  /-
    🎉 no goals
  -/


theorem sInf_adj_of_nonempty {s : Set (SimpleGraph V)} (hs : s.Nonempty) :
    (sInf s).Adj a b ↔ ∀ G ∈ s, Adj G a b :=
  sInf_adj.trans <|
    and_iff_left_of_imp <| by
      /-
        V : Type u
        a b : V
        s : Set (SimpleGraph V)
        hs : s.Nonempty
        ⊢ (∀ (G : SimpleGraph V), Membership.mem s G → G.Adj a b) → Ne a b
      -/
      obtain ⟨G, hG⟩ := hs
      /-
        case intro
        V : Type u
        a b : V
        s : Set (SimpleGraph V)
        G : SimpleGraph V
        hG : Membership.mem s G
        ⊢ (∀ (G : SimpleGraph V), Membership.mem s G → G.Adj a b) → Ne a b
      -/
      exact fun h => (h _ hG).ne
      /-
        🎉 no goals
      -/


theorem iInf_adj_of_nonempty [Nonempty ι] {f : ι → SimpleGraph V} :
    (⨅ i, f i).Adj a b ↔ ∀ i, (f i).Adj a b := by
  /-
    ι : Sort u_1
    V : Type u
    a b : V
    inst✝ : Nonempty ι
    f : ι → SimpleGraph V
    ⊢ Iff ((iInf fun i => f i).Adj a b) (∀ (i : ι), (f i).Adj a b)
  -/
  rw [iInf, sInf_adj_of_nonempty (Set.range_nonempty _), Set.forall_mem_range]
  /-
    🎉 no goals
  -/


/-- For graphs `G`, `H`, `G ≤ H` iff `∀ a b, G.Adj a b → H.Adj a b`. -/
instance distribLattice : DistribLattice (SimpleGraph V) :=
  { show DistribLattice (SimpleGraph V) from
      adj_injective.distribLattice _ (fun _ _ => rfl) fun _ _ => rfl with
    le := fun G H => ∀ ⦃a b⦄, G.Adj a b → H.Adj a b }


instance completeAtomicBooleanAlgebra : CompleteAtomicBooleanAlgebra (SimpleGraph V) :=
  { SimpleGraph.distribLattice with
    le := (· ≤ ·)
    sup := (· ⊔ ·)
    inf := (· ⊓ ·)
    compl := HasCompl.compl
    sdiff := (· \ ·)
    top := completeGraph V
    bot := emptyGraph V
    le_top := fun x _ _ h => x.ne_of_adj h
    bot_le := fun _ _ _ h => h.elim
    sdiff_eq := fun x y => by
      /-
        ι : Sort u_1
        V : Type u
        G : SimpleGraph V
        a b c u v w : V
        e : Sym2 V
        x y : SimpleGraph V
        ⊢ Eq (SDiff.sdiff x y) (Min.min x (HasCompl.compl y))
      -/
      ext v w
      /-
        case Adj.h.h.a
        ι : Sort u_1
        V : Type u
        G : SimpleGraph V
        a b c u v✝ w✝ : V
        e : Sym2 V
        x y : SimpleGraph V
        v w : V
        ⊢ Iff ((SDiff.sdiff x y).Adj v w) ((Min.min x (HasCompl.compl y)).Adj v w)
      -/
      refine ⟨fun h => ⟨h.1, ⟨?_, h.2⟩⟩, fun h => ⟨h.1, h.2.2⟩⟩
      /-
        case Adj.h.h.a
        ι : Sort u_1
        V : Type u
        G : SimpleGraph V
        a b c u v✝ w✝ : V
        e : Sym2 V
        x y : SimpleGraph V
        v w : V
        h : (SDiff.sdiff x y).Adj v w
        ⊢ Ne v w
      -/
      rintro rfl
      /-
        ι : Sort u_1
        V : Type u
        G✝ : SimpleGraph V
        a b c u v✝ w✝ : V
        e : Sym2 V
        G : SimpleGraph V
        v w : V
        hvw : Top.top.Adj v w
        ⊢ (Max.max G (HasCompl.compl G)).Adj v w
      -/
      /-
        case Adj.h.h.a
        ι : Sort u_1
        V : Type u
        G : SimpleGraph V
        a b c u v✝ w : V
        e : Sym2 V
        x y : SimpleGraph V
        v : V
        h : (SDiff.sdiff x y).Adj v v
        ⊢ False
      -/
        /-
          case pos
          ι : Sort u_1
          V : Type u
          G✝ : SimpleGraph V
          a b c u v✝ w✝ : V
          e : Sym2 V
          G : SimpleGraph V
          v w : V
          hvw : Top.top.Adj v w
          h : G.Adj v w
          ⊢ (Max.max G (HasCompl.compl G)).Adj v w
        -/
      exact x.irrefl h.1
        /-
          🎉 no goals
        -/
        /-
          case neg
          ι : Sort u_1
          V : Type u
          G✝ : SimpleGraph V
          a b c u v✝ w✝ : V
          e : Sym2 V
          G : SimpleGraph V
          v w : V
          hvw : Top.top.Adj v w
          h : Not (G.Adj v w)
          ⊢ (Max.max G (HasCompl.compl G)).Adj v w
        -/
      /-
        ι : Sort u_1
        V : Type u
        G✝ : SimpleGraph V
        a✝ b✝ c u v w : V
        e : Sym2 V
        s : Set (SimpleGraph V)
        G : SimpleGraph V
        hG : ∀ (b : SimpleGraph V), Membership.mem s b → LE.le b G
        a b : V
        ⊢ (SupSet.sSup s).Adj a b → G.Adj a b
      -/
      /-
        🎉 no goals
      -/
      /-
        case intro.intro
        ι : Sort u_1
        V : Type u
        G✝ : SimpleGraph V
        a✝ b✝ c u v w : V
        e : Sym2 V
        s : Set (SimpleGraph V)
        G : SimpleGraph V
        hG : ∀ (b : SimpleGraph V), Membership.mem s b → LE.le b G
        a b : V
        H : SimpleGraph V
        hH : Membership.mem s H
        hab : H.Adj a b
        ⊢ G.Adj a b
      -/
        /-
          🎉 no goals
        -/
      /-
        🎉 no goals
      -/
    inf_compl_le_bot := fun _ _ _ h => False.elim <| h.2.2 h.1
    top_le_sup_compl := fun G v w hvw => by
      by_cases h : G.Adj v w
      · exact Or.inl h
      · exact Or.inr ⟨hvw, h⟩
    sSup := sSup
    le_sSup := fun _ G hG _ _ hab => ⟨G, hG, hab⟩
    sSup_le := fun s G hG a b => by
      rintro ⟨H, hH, hab⟩
      exact hG _ hH hab
    sInf := sInf
    sInf_le := fun _ _ hG _ _ hab => hab.1 hG
    le_sInf := fun _ _ hG _ _ hab => ⟨fun _ hH => hG _ hH hab, hab.ne⟩
                                /-
                                  ι : Sort u_1
                                  V : Type u
                                  G : SimpleGraph V
                                  a b c u v w : V
                                  e : Sym2 V
                                  ι✝ : Type u
                                  κ✝ : ι✝ → Type u
                                  f : (a : ι✝) → κ✝ a → SimpleGraph V
                                  ⊢ Eq (iInf fun a => iSup fun b => f a b) (iSup fun g => iInf fun a => f a (g a))
                                -/
    iInf_iSup_eq := fun f => by ext; simp [Classical.skolem] }
                                     /-
                                       🎉 no goals
                                     -/


@[simp]
theorem top_adj (v w : V) : (⊤ : SimpleGraph V).Adj v w ↔ v ≠ w :=
  Iff.rfl


@[simp]
theorem bot_adj (v w : V) : (⊥ : SimpleGraph V).Adj v w ↔ False :=
  Iff.rfl


@[simp]
theorem completeGraph_eq_top (V : Type u) : completeGraph V = ⊤ :=
  rfl


@[simp]
theorem emptyGraph_eq_bot (V : Type u) : emptyGraph V = ⊥ :=
  rfl


@[simps]
instance (V : Type u) : Inhabited (SimpleGraph V) :=
  ⟨⊥⟩


instance [Subsingleton V] : Unique (SimpleGraph V) where
  default := ⊥
               /-
                 ι : Sort u_1
                 V : Type u
                 G✝ : SimpleGraph V
                 a b c u v w : V
                 e : Sym2 V
                 inst✝ : Subsingleton V
                 G : SimpleGraph V
                 ⊢ Eq G Inhabited.default
               -/
  uniq G := by ext a b; have := Subsingleton.elim a b; simp [this]
                                                       /-
                                                         🎉 no goals
                                                       -/


instance [Nontrivial V] : Nontrivial (SimpleGraph V) :=
  ⟨⟨⊥, ⊤, fun h ↦ not_subsingleton V ⟨by simpa only [← adj_inj, funext_iff, bot_adj,
    top_adj, ne_eq, eq_iff_iff, false_iff, not_not] using h⟩⟩⟩


instance Bot.adjDecidable : DecidableRel (⊥ : SimpleGraph V).Adj :=
  inferInstanceAs <| DecidableRel fun _ _ => False


instance Sup.adjDecidable : DecidableRel (G ⊔ H).Adj :=
  inferInstanceAs <| DecidableRel fun v w => G.Adj v w ∨ H.Adj v w


instance Inf.adjDecidable : DecidableRel (G ⊓ H).Adj :=
  inferInstanceAs <| DecidableRel fun v w => G.Adj v w ∧ H.Adj v w


instance Sdiff.adjDecidable : DecidableRel (G \ H).Adj :=
  inferInstanceAs <| DecidableRel fun v w => G.Adj v w ∧ ¬H.Adj v w


instance Top.adjDecidable : DecidableRel (⊤ : SimpleGraph V).Adj :=
  inferInstanceAs <| DecidableRel fun v w => v ≠ w


instance Compl.adjDecidable : DecidableRel (Gᶜ.Adj) :=
  inferInstanceAs <| DecidableRel fun v w => v ≠ w ∧ ¬G.Adj v w


/-- `G.support` is the set of vertices that form edges in `G`. -/
def support : Set V :=
  Rel.dom G.Adj


theorem mem_support {v : V} : v ∈ G.support ↔ ∃ w, G.Adj v w :=
  Iff.rfl


theorem support_mono {G G' : SimpleGraph V} (h : G ≤ G') : G.support ⊆ G'.support :=
  Rel.dom_mono h


/-- `G.neighborSet v` is the set of vertices adjacent to `v` in `G`. -/
def neighborSet (v : V) : Set V := {w | G.Adj v w}


instance neighborSet.memDecidable (v : V) [DecidableRel G.Adj] :
    DecidablePred (· ∈ G.neighborSet v) :=
  inferInstanceAs <| DecidablePred (Adj G v)


/-- The edges of G consist of the unordered pairs of vertices related by
`G.Adj`. This is the order embedding; for the edge set of a particular graph, see
`SimpleGraph.edgeSet`.

The way `edgeSet` is defined is such that `mem_edgeSet` is proved by `Iff.rfl`.
(That is, `s(v, w) ∈ G.edgeSet` is definitionally equal to `G.Adj v w`.)
-/
-- Porting note: We need a separate definition so that dot notation works.
def edgeSetEmbedding (V : Type*) : SimpleGraph V ↪o Set (Sym2 V) :=
  OrderEmbedding.ofMapLEIff (fun G => Sym2.fromRel G.symm) fun _ _ =>
    ⟨fun h a b => @h s(a, b), fun h e => Sym2.ind @h e⟩


/-- `G.edgeSet` is the edge set for `G`.
This is an abbreviation for `edgeSetEmbedding G` that permits dot notation. -/
abbrev edgeSet (G : SimpleGraph V) : Set (Sym2 V) := edgeSetEmbedding V G


@[simp]
theorem mem_edgeSet : s(v, w) ∈ G.edgeSet ↔ G.Adj v w :=
  Iff.rfl


theorem not_isDiag_of_mem_edgeSet : e ∈ edgeSet G → ¬e.IsDiag :=
  Sym2.ind (fun _ _ => Adj.ne) e


theorem edgeSet_inj : G₁.edgeSet = G₂.edgeSet ↔ G₁ = G₂ := (edgeSetEmbedding V).eq_iff_eq


@[simp]
theorem edgeSet_subset_edgeSet : edgeSet G₁ ⊆ edgeSet G₂ ↔ G₁ ≤ G₂ :=
  (edgeSetEmbedding V).le_iff_le


@[simp]
theorem edgeSet_ssubset_edgeSet : edgeSet G₁ ⊂ edgeSet G₂ ↔ G₁ < G₂ :=
  (edgeSetEmbedding V).lt_iff_lt


theorem edgeSet_injective : Injective (edgeSet : SimpleGraph V → Set (Sym2 V)) :=
  (edgeSetEmbedding V).injective


alias ⟨_, edgeSet_mono⟩ := edgeSet_subset_edgeSet


alias ⟨_, edgeSet_strict_mono⟩ := edgeSet_ssubset_edgeSet


@[simp]
theorem edgeSet_bot : (⊥ : SimpleGraph V).edgeSet = ∅ :=
  Sym2.fromRel_bot


@[simp]
theorem edgeSet_top : (⊤ : SimpleGraph V).edgeSet = {e | ¬e.IsDiag} :=
  Sym2.fromRel_ne


@[simp]
theorem edgeSet_subset_setOf_not_isDiag : G.edgeSet ⊆ {e | ¬e.IsDiag} :=
  fun _ h => (Sym2.fromRel_irreflexive (sym := G.symm)).mp G.loopless h


@[simp]
theorem edgeSet_sup : (G₁ ⊔ G₂).edgeSet = G₁.edgeSet ∪ G₂.edgeSet := by
  /-
    V : Type u
    G₁ G₂ : SimpleGraph V
    ⊢ Eq (Max.max G₁ G₂).edgeSet (Union.union G₁.edgeSet G₂.edgeSet)
  -/
  ext ⟨x, y⟩
  /-
    case h.mk.mk
    V : Type u
    G₁ G₂ : SimpleGraph V
    x✝ : Sym2 V
    x y : V
    ⊢ Iff (Membership.mem (Max.max G₁ G₂).edgeSet (Quot.mk (Sym2.Rel V) { fst := x …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem edgeSet_inf : (G₁ ⊓ G₂).edgeSet = G₁.edgeSet ∩ G₂.edgeSet := by
  /-
    V : Type u
    G₁ G₂ : SimpleGraph V
    ⊢ Eq (Min.min G₁ G₂).edgeSet (Inter.inter G₁.edgeSet G₂.edgeSet)
  -/
  ext ⟨x, y⟩
  /-
    case h.mk.mk
    V : Type u
    G₁ G₂ : SimpleGraph V
    x✝ : Sym2 V
    x y : V
    ⊢ Iff (Membership.mem (Min.min G₁ G₂).edgeSet (Quot.mk (Sym2.Rel V) { fst := x …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem edgeSet_sdiff : (G₁ \ G₂).edgeSet = G₁.edgeSet \ G₂.edgeSet := by
  /-
    V : Type u
    G₁ G₂ : SimpleGraph V
    ⊢ Eq (SDiff.sdiff G₁ G₂).edgeSet (SDiff.sdiff G₁.edgeSet G₂.edgeSet)
  -/
  ext ⟨x, y⟩
  /-
    case h.mk.mk
    V : Type u
    G₁ G₂ : SimpleGraph V
    x✝ : Sym2 V
    x y : V
    ⊢ Iff (Membership.mem (SDiff.sdiff G₁ G₂).edgeSet (Quot.mk (Sym2.Rel V) { fst  …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp] lemma disjoint_edgeSet : Disjoint G₁.edgeSet G₂.edgeSet ↔ Disjoint G₁ G₂ := by
  rw [Set.disjoint_iff, disjoint_iff_inf_le, ← edgeSet_inf, ← edgeSet_bot, ← Set.le_iff_subset,
    OrderEmbedding.le_iff_le]


                                                             /-
                                                               V : Type u
                                                               G : SimpleGraph V
                                                               ⊢ Iff (Eq G.edgeSet EmptyCollection.emptyCollection) (Eq G Bot.bot)
                                                             -/
@[simp] lemma edgeSet_eq_empty : G.edgeSet = ∅ ↔ G = ⊥ := by rw [← edgeSet_bot, edgeSet_inj]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp] lemma edgeSet_nonempty : G.edgeSet.Nonempty ↔ G ≠ ⊥ := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Iff G.edgeSet.Nonempty (Ne G Bot.bot)
  -/
  rw [Set.nonempty_iff_ne_empty, edgeSet_eq_empty.ne]
  /-
    🎉 no goals
  -/


/-- This lemma, combined with `edgeSet_sdiff` and `edgeSet_from_edgeSet`,
allows proving `(G \ from_edgeSet s).edge_set = G.edgeSet \ s` by `simp`. -/
@[simp]
theorem edgeSet_sdiff_sdiff_isDiag (G : SimpleGraph V) (s : Set (Sym2 V)) :
    G.edgeSet \ (s \ { e | e.IsDiag }) = G.edgeSet \ s := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    ⊢ Eq (SDiff.sdiff G.edgeSet (SDiff.sdiff s (setOf fun e => e.IsDiag))) (SDiff. …
  -/
  ext e
  /-
    case h
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    e : Sym2 V
    ⊢ Iff (Membership.mem (SDiff.sdiff G.edgeSet (SDiff.sdiff s (setOf fun e => e. …
  -/
  simp only [Set.mem_diff, Set.mem_setOf_eq, not_and, not_not, and_congr_right_iff]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    e : Sym2 V
    ⊢ Membership.mem G.edgeSet e → Iff (Membership.mem s e → e.IsDiag) (Not (Membe …
  -/
  intro h
  /-
    case h
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    e : Sym2 V
    h : Membership.mem G.edgeSet e
    ⊢ Iff (Membership.mem s e → e.IsDiag) (Not (Membership.mem s e))
  -/
  simp only [G.not_isDiag_of_mem_edgeSet h, imp_false]
  /-
    🎉 no goals
  -/


/-- Two vertices are adjacent iff there is an edge between them. The
condition `v ≠ w` ensures they are different endpoints of the edge,
which is necessary since when `v = w` the existential
`∃ (e ∈ G.edgeSet), v ∈ e ∧ w ∈ e` is satisfied by every edge
incident to `v`. -/
theorem adj_iff_exists_edge {v w : V} : G.Adj v w ↔ v ≠ w ∧ ∃ e ∈ G.edgeSet, v ∈ e ∧ w ∈ e := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff (G.Adj v w) (And (Ne v w) (Exists fun e => And (Membership.mem G.edgeSet …
  -/
  refine ⟨fun _ => ⟨G.ne_of_adj ‹_›, s(v, w), by simpa⟩, ?_⟩
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ And (Ne v w) (Exists fun e => And (Membership.mem G.edgeSet e) (And (Members …
  -/
  rintro ⟨hne, e, he, hv⟩
  /-
    case intro.intro.intro
    V : Type u
    G : SimpleGraph V
    v w : V
    hne : Ne v w
    e : Sym2 V
    he : Membership.mem G.edgeSet e
    hv : And (Membership.mem e v) (Membership.mem e w)
    ⊢ G.Adj v w
  -/
  rw [Sym2.mem_and_mem_iff hne] at hv
  /-
    case intro.intro.intro
    V : Type u
    G : SimpleGraph V
    v w : V
    hne : Ne v w
    e : Sym2 V
    he : Membership.mem G.edgeSet e
    hv : Eq e (Sym2.mk { fst := v, snd := w })
    ⊢ G.Adj v w
  -/
  subst e
  /-
    case intro.intro.intro
    V : Type u
    G : SimpleGraph V
    v w : V
    hne : Ne v w
    he : Membership.mem G.edgeSet (Sym2.mk { fst := v, snd := w })
    ⊢ G.Adj v w
  -/
  rwa [mem_edgeSet] at he
  /-
    🎉 no goals
  -/


theorem adj_iff_exists_edge_coe : G.Adj a b ↔ ∃ e : G.edgeSet, e.val = s(a, b) := by
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    ⊢ Iff (G.Adj a b) (Exists fun e => Eq (↑e) (Sym2.mk { fst := a, snd := b }))
  -/
  simp only [mem_edgeSet, exists_prop, SetCoe.exists, exists_eq_right, Subtype.coe_mk]
  /-
    🎉 no goals
  -/


theorem edge_other_ne {e : Sym2 V} (he : e ∈ G.edgeSet) {v : V} (h : v ∈ e) :
    Sym2.Mem.other h ≠ v := by
  /-
    V : Type u
    G : SimpleGraph V
    e : Sym2 V
    he : Membership.mem G.edgeSet e
    v : V
    h : Membership.mem e v
    ⊢ Ne (Sym2.Mem.other h) v
  -/
  rw [← Sym2.other_spec h, Sym2.eq_swap] at he
  /-
    V : Type u
    G : SimpleGraph V
    e : Sym2 V
    v : V
    h : Membership.mem e v
    he : Membership.mem G.edgeSet (Sym2.mk { fst := Sym2.Mem.other h, snd := v })
    ⊢ Ne (Sym2.Mem.other h) v
  -/
  exact G.ne_of_adj he
  /-
    🎉 no goals
  -/


instance decidableMemEdgeSet [DecidableRel G.Adj] : DecidablePred (· ∈ G.edgeSet) :=
  Sym2.fromRel.decidablePred G.symm


instance fintypeEdgeSet [Fintype (Sym2 V)] [DecidableRel G.Adj] : Fintype G.edgeSet :=
  Subtype.fintype _


instance fintypeEdgeSetBot : Fintype (⊥ : SimpleGraph V).edgeSet := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    G₁ G₂ : SimpleGraph V
    ⊢ Fintype ↑Bot.bot.edgeSet
  -/
  rw [edgeSet_bot]
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    G₁ G₂ : SimpleGraph V
    ⊢ Fintype ↑EmptyCollection.emptyCollection
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance fintypeEdgeSetSup [DecidableEq V] [Fintype G₁.edgeSet] [Fintype G₂.edgeSet] :
    Fintype (G₁ ⊔ G₂).edgeSet := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    G₁ G₂ : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype ↑G₁.edgeSet
    inst✝ : Fintype ↑G₂.edgeSet
    ⊢ Fintype ↑(Max.max G₁ G₂).edgeSet
  -/
  rw [edgeSet_sup]
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    G₁ G₂ : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype ↑G₁.edgeSet
    inst✝ : Fintype ↑G₂.edgeSet
    ⊢ Fintype ↑(Union.union G₁.edgeSet G₂.edgeSet)
  -/
  infer_instance
  /-
    🎉 no goals
  -/


instance fintypeEdgeSetInf [DecidableEq V] [Fintype G₁.edgeSet] [Fintype G₂.edgeSet] :
    Fintype (G₁ ⊓ G₂).edgeSet := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    G₁ G₂ : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype ↑G₁.edgeSet
    inst✝ : Fintype ↑G₂.edgeSet
    ⊢ Fintype ↑(Min.min G₁ G₂).edgeSet
  -/
  rw [edgeSet_inf]
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    G₁ G₂ : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype ↑G₁.edgeSet
    inst✝ : Fintype ↑G₂.edgeSet
    ⊢ Fintype ↑(Inter.inter G₁.edgeSet G₂.edgeSet)
  -/
  exact Set.fintypeInter _ _
  /-
    🎉 no goals
  -/


instance fintypeEdgeSetSdiff [DecidableEq V] [Fintype G₁.edgeSet] [Fintype G₂.edgeSet] :
    Fintype (G₁ \ G₂).edgeSet := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    G₁ G₂ : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype ↑G₁.edgeSet
    inst✝ : Fintype ↑G₂.edgeSet
    ⊢ Fintype ↑(SDiff.sdiff G₁ G₂).edgeSet
  -/
  rw [edgeSet_sdiff]
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    G₁ G₂ : SimpleGraph V
    inst✝² : DecidableEq V
    inst✝¹ : Fintype ↑G₁.edgeSet
    inst✝ : Fintype ↑G₂.edgeSet
    ⊢ Fintype ↑(SDiff.sdiff G₁.edgeSet G₂.edgeSet)
  -/
  exact Set.fintypeDiff _ _
  /-
    🎉 no goals
  -/


/-- `fromEdgeSet` constructs a `SimpleGraph` from a set of edges, without loops. -/
def fromEdgeSet : SimpleGraph V where
  Adj := Sym2.ToRel s ⊓ Ne
  symm _ _ h := ⟨Sym2.toRel_symmetric s h.1, h.2.symm⟩


@[simp]
theorem fromEdgeSet_adj : (fromEdgeSet s).Adj v w ↔ s(v, w) ∈ s ∧ v ≠ w :=
  Iff.rfl

-- Note: we need to make sure `fromEdgeSet_adj` and this lemma are confluent.
-- In particular, both yield `s(u, v) ∈ (fromEdgeSet s).edgeSet` ==> `s(v, w) ∈ s ∧ v ≠ w`.

@[simp]
theorem edgeSet_fromEdgeSet : (fromEdgeSet s).edgeSet = s \ { e | e.IsDiag } := by
  /-
    V : Type u
    s : Set (Sym2 V)
    ⊢ Eq (SimpleGraph.fromEdgeSet s).edgeSet (SDiff.sdiff s (setOf fun e => e.IsDi …
  -/
  ext e
  /-
    case h
    V : Type u
    s : Set (Sym2 V)
    e : Sym2 V
    ⊢ Iff (Membership.mem (SimpleGraph.fromEdgeSet s).edgeSet e) (Membership.mem ( …
  -/
  exact Sym2.ind (by simp) e
  /-
    🎉 no goals
  -/


@[simp]
theorem fromEdgeSet_edgeSet : fromEdgeSet G.edgeSet = G := by
  /-
    V : Type u
    G : SimpleGraph V
    ⊢ Eq (SimpleGraph.fromEdgeSet G.edgeSet) G
  -/
  ext v w
  /-
    case Adj.h.h.a
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff ((SimpleGraph.fromEdgeSet G.edgeSet).Adj v w) (G.Adj v w)
  -/
  exact ⟨fun h => h.1, fun h => ⟨h, G.ne_of_adj h⟩⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem fromEdgeSet_empty : fromEdgeSet (∅ : Set (Sym2 V)) = ⊥ := by
  /-
    V : Type u
    ⊢ Eq (SimpleGraph.fromEdgeSet EmptyCollection.emptyCollection) Bot.bot
  -/
  ext v w
  /-
    case Adj.h.h.a
    V : Type u
    v w : V
    ⊢ Iff ((SimpleGraph.fromEdgeSet EmptyCollection.emptyCollection).Adj v w) (Bot …
  -/
  simp only [fromEdgeSet_adj, Set.mem_empty_iff_false, false_and, bot_adj]
  /-
    🎉 no goals
  -/


@[simp]
theorem fromEdgeSet_univ : fromEdgeSet (Set.univ : Set (Sym2 V)) = ⊤ := by
  /-
    V : Type u
    ⊢ Eq (SimpleGraph.fromEdgeSet Set.univ) Top.top
  -/
  ext v w
  /-
    case Adj.h.h.a
    V : Type u
    v w : V
    ⊢ Iff ((SimpleGraph.fromEdgeSet Set.univ).Adj v w) (Top.top.Adj v w)
  -/
  simp only [fromEdgeSet_adj, Set.mem_univ, true_and, top_adj]
  /-
    🎉 no goals
  -/


@[simp]
theorem fromEdgeSet_inter (s t : Set (Sym2 V)) :
    fromEdgeSet (s ∩ t) = fromEdgeSet s ⊓ fromEdgeSet t := by
  /-
    V : Type u
    s t : Set (Sym2 V)
    ⊢ Eq (SimpleGraph.fromEdgeSet (Inter.inter s t)) (Min.min (SimpleGraph.fromEdg …
  -/
  ext v w
  /-
    case Adj.h.h.a
    V : Type u
    s t : Set (Sym2 V)
    v w : V
    ⊢ Iff ((SimpleGraph.fromEdgeSet (Inter.inter s t)).Adj v w) ((Min.min (SimpleG …
  -/
  simp only [fromEdgeSet_adj, Set.mem_inter_iff, Ne, inf_adj]
  /-
    case Adj.h.h.a
    V : Type u
    s t : Set (Sym2 V)
    v w : V
    ⊢ Iff (And (And (Membership.mem s (Sym2.mk { fst := v, snd := w })) (Membershi …
  -/
  tauto
  /-
    🎉 no goals
  -/


@[simp]
theorem fromEdgeSet_union (s t : Set (Sym2 V)) :
    fromEdgeSet (s ∪ t) = fromEdgeSet s ⊔ fromEdgeSet t := by
  /-
    V : Type u
    s t : Set (Sym2 V)
    ⊢ Eq (SimpleGraph.fromEdgeSet (Union.union s t)) (Max.max (SimpleGraph.fromEdg …
  -/
  ext v w
  /-
    case Adj.h.h.a
    V : Type u
    s t : Set (Sym2 V)
    v w : V
    ⊢ Iff ((SimpleGraph.fromEdgeSet (Union.union s t)).Adj v w) ((Max.max (SimpleG …
  -/
  simp [Set.mem_union, or_and_right]
  /-
    🎉 no goals
  -/


@[simp]
theorem fromEdgeSet_sdiff (s t : Set (Sym2 V)) :
    fromEdgeSet (s \ t) = fromEdgeSet s \ fromEdgeSet t := by
  /-
    V : Type u
    s t : Set (Sym2 V)
    ⊢ Eq (SimpleGraph.fromEdgeSet (SDiff.sdiff s t)) (SDiff.sdiff (SimpleGraph.fro …
  -/
  ext v w
  /-
    case Adj.h.h.a
    V : Type u
    s t : Set (Sym2 V)
    v w : V
    ⊢ Iff ((SimpleGraph.fromEdgeSet (SDiff.sdiff s t)).Adj v w) ((SDiff.sdiff (Sim …
  -/
                  /-
                    🎉 no goals
                  -/
  constructor <;> simp +contextual
                  /-
                    🎉 no goals
                  -/


@[gcongr, mono]
theorem fromEdgeSet_mono {s t : Set (Sym2 V)} (h : s ⊆ t) : fromEdgeSet s ≤ fromEdgeSet t := by
  /-
    V : Type u
    s t : Set (Sym2 V)
    h : HasSubset.Subset s t
    ⊢ LE.le (SimpleGraph.fromEdgeSet s) (SimpleGraph.fromEdgeSet t)
  -/
  rintro v w
  simp +contextual only [fromEdgeSet_adj, Ne, not_false_iff,
    and_true, and_imp]
  /-
    V : Type u
    s t : Set (Sym2 V)
    h : HasSubset.Subset s t
    v w : V
    ⊢ Membership.mem s (Sym2.mk { fst := v, snd := w }) → Not (Eq v w) → Membershi …
  -/
  exact fun vws _ => h vws
  /-
    🎉 no goals
  -/


@[simp] lemma disjoint_fromEdgeSet : Disjoint G (fromEdgeSet s) ↔ Disjoint G.edgeSet s := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    ⊢ Iff (Disjoint G (SimpleGraph.fromEdgeSet s)) (Disjoint G.edgeSet s)
  -/
  conv_rhs => rw [← Set.diff_union_inter s {e : Sym2 V | e.IsDiag}]
  /-
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    ⊢ Iff (Disjoint G (SimpleGraph.fromEdgeSet s)) (Disjoint G.edgeSet (Union.unio …
  -/
  rw [← disjoint_edgeSet,  edgeSet_fromEdgeSet, Set.disjoint_union_right, and_iff_left]
  /-
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    ⊢ Disjoint G.edgeSet (Inter.inter s (setOf fun e => e.IsDiag))
  -/
  exact Set.disjoint_left.2 fun e he he' ↦ not_isDiag_of_mem_edgeSet _ he he'.2
  /-
    🎉 no goals
  -/


@[simp] lemma fromEdgeSet_disjoint : Disjoint (fromEdgeSet s) G ↔ Disjoint s G.edgeSet := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    ⊢ Iff (Disjoint (SimpleGraph.fromEdgeSet s) G) (Disjoint s G.edgeSet)
  -/
  rw [disjoint_comm, disjoint_fromEdgeSet, disjoint_comm]
  /-
    🎉 no goals
  -/


instance [DecidableEq V] [Fintype s] : Fintype (fromEdgeSet s).edgeSet := by
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    s : Set (Sym2 V)
    inst✝¹ : DecidableEq V
    inst✝ : Fintype ↑s
    ⊢ Fintype ↑(SimpleGraph.fromEdgeSet s).edgeSet
  -/
  rw [edgeSet_fromEdgeSet s]
  /-
    ι : Sort u_1
    V : Type u
    G : SimpleGraph V
    a b c u v w : V
    e : Sym2 V
    s : Set (Sym2 V)
    inst✝¹ : DecidableEq V
    inst✝ : Fintype ↑s
    ⊢ Fintype ↑(SDiff.sdiff s (setOf fun e => e.IsDiag))
  -/
  infer_instance
  /-
    🎉 no goals
  -/


/-- Set of edges incident to a given vertex, aka incidence set. -/
def incidenceSet (v : V) : Set (Sym2 V) :=
  { e ∈ G.edgeSet | v ∈ e }


theorem incidenceSet_subset (v : V) : G.incidenceSet v ⊆ G.edgeSet := fun _ h => h.1


theorem mk'_mem_incidenceSet_iff : s(b, c) ∈ G.incidenceSet a ↔ G.Adj b c ∧ (a = b ∨ a = c) :=
  and_congr_right' Sym2.mem_iff


theorem mk'_mem_incidenceSet_left_iff : s(a, b) ∈ G.incidenceSet a ↔ G.Adj a b :=
  and_iff_left <| Sym2.mem_mk_left _ _


theorem mk'_mem_incidenceSet_right_iff : s(a, b) ∈ G.incidenceSet b ↔ G.Adj a b :=
  and_iff_left <| Sym2.mem_mk_right _ _


theorem edge_mem_incidenceSet_iff {e : G.edgeSet} : ↑e ∈ G.incidenceSet a ↔ a ∈ (e : Sym2 V) :=
  and_iff_right e.2


theorem incidenceSet_inter_incidenceSet_subset (h : a ≠ b) :
    G.incidenceSet a ∩ G.incidenceSet b ⊆ {s(a, b)} := fun _e he =>
  (Sym2.mem_and_mem_iff h).1 ⟨he.1.2, he.2.2⟩


theorem incidenceSet_inter_incidenceSet_of_adj (h : G.Adj a b) :
    G.incidenceSet a ∩ G.incidenceSet b = {s(a, b)} := by
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    h : G.Adj a b
    ⊢ Eq (Inter.inter (G.incidenceSet a) (G.incidenceSet b)) (Singleton.singleton  …
  -/
  refine (G.incidenceSet_inter_incidenceSet_subset <| h.ne).antisymm ?_
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    h : G.Adj a b
    ⊢ HasSubset.Subset (Singleton.singleton (Sym2.mk { fst := a, snd := b })) (Int …
  -/
  rintro _ (rfl : _ = s(a, b))
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    h : G.Adj a b
    ⊢ Membership.mem (Inter.inter (G.incidenceSet a) (G.incidenceSet b)) (Sym2.mk  …
  -/
  exact ⟨G.mk'_mem_incidenceSet_left_iff.2 h, G.mk'_mem_incidenceSet_right_iff.2 h⟩
  /-
    🎉 no goals
  -/


theorem adj_of_mem_incidenceSet (h : a ≠ b) (ha : e ∈ G.incidenceSet a)
    (hb : e ∈ G.incidenceSet b) : G.Adj a b := by
  rwa [← mk'_mem_incidenceSet_left_iff, ←
    Set.mem_singleton_iff.1 <| G.incidenceSet_inter_incidenceSet_subset h ⟨ha, hb⟩]


theorem incidenceSet_inter_incidenceSet_of_not_adj (h : ¬G.Adj a b) (hn : a ≠ b) :
    G.incidenceSet a ∩ G.incidenceSet b = ∅ := by
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    h : Not (G.Adj a b)
    hn : Ne a b
    ⊢ Eq (Inter.inter (G.incidenceSet a) (G.incidenceSet b)) EmptyCollection.empty …
  -/
  simp_rw [Set.eq_empty_iff_forall_not_mem, Set.mem_inter_iff, not_and]
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    h : Not (G.Adj a b)
    hn : Ne a b
    ⊢ ∀ (x : Sym2 V), Membership.mem (G.incidenceSet a) x → Not (Membership.mem (G …
  -/
  intro u ha hb
  /-
    V : Type u
    G : SimpleGraph V
    a b : V
    h : Not (G.Adj a b)
    hn : Ne a b
    u : Sym2 V
    ha : Membership.mem (G.incidenceSet a) u
    hb : Membership.mem (G.incidenceSet b) u
    ⊢ False
  -/
  exact h (G.adj_of_mem_incidenceSet hn ha hb)
  /-
    🎉 no goals
  -/


instance decidableMemIncidenceSet [DecidableEq V] [DecidableRel G.Adj] (v : V) :
    DecidablePred (· ∈ G.incidenceSet v) :=
  inferInstanceAs <| DecidablePred fun e => e ∈ G.edgeSet ∧ v ∈ e


@[simp]
theorem mem_neighborSet (v w : V) : w ∈ G.neighborSet v ↔ G.Adj v w :=
  Iff.rfl


                                                           /-
                                                             V : Type u
                                                             G : SimpleGraph V
                                                             a : V
                                                             ⊢ Not (Membership.mem (G.neighborSet a) a)
                                                           -/
lemma not_mem_neighborSet_self : a ∉ G.neighborSet a := by simp
                                                           /-
                                                             🎉 no goals
                                                           -/


@[simp]
theorem mem_incidenceSet (v w : V) : s(v, w) ∈ G.incidenceSet v ↔ G.Adj v w := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff (Membership.mem (G.incidenceSet v) (Sym2.mk { fst := v, snd := w })) (G. …
  -/
  simp [incidenceSet]
  /-
    🎉 no goals
  -/


theorem mem_incidence_iff_neighbor {v w : V} :
    s(v, w) ∈ G.incidenceSet v ↔ w ∈ G.neighborSet v := by
  /-
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff (Membership.mem (G.incidenceSet v) (Sym2.mk { fst := v, snd := w })) (Me …
  -/
  simp only [mem_incidenceSet, mem_neighborSet]
  /-
    🎉 no goals
  -/


theorem adj_incidenceSet_inter {v : V} {e : Sym2 V} (he : e ∈ G.edgeSet) (h : v ∈ e) :
    G.incidenceSet v ∩ G.incidenceSet (Sym2.Mem.other h) = {e} := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    e : Sym2 V
    he : Membership.mem G.edgeSet e
    h : Membership.mem e v
    ⊢ Eq (Inter.inter (G.incidenceSet v) (G.incidenceSet (Sym2.Mem.other h))) (Sin …
  -/
  ext e'
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v : V
    e : Sym2 V
    he : Membership.mem G.edgeSet e
    h : Membership.mem e v
    e' : Sym2 V
    ⊢ Iff (Membership.mem (Inter.inter (G.incidenceSet v) (G.incidenceSet (Sym2.Me …
  -/
  simp only [incidenceSet, Set.mem_sep_iff, Set.mem_inter_iff, Set.mem_singleton_iff]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v : V
    e : Sym2 V
    he : Membership.mem G.edgeSet e
    h : Membership.mem e v
    e' : Sym2 V
    ⊢ Iff (And (And (Membership.mem G.edgeSet e') (Membership.mem e' v)) (And (Mem …
  -/
  refine ⟨fun h' => ?_, ?_⟩
    /-
      case h.refine_1
      V : Type u
      G : SimpleGraph V
      v : V
      e : Sym2 V
      he : Membership.mem G.edgeSet e
      h : Membership.mem e v
      e' : Sym2 V
      h' : And (And (Membership.mem G.edgeSet e') (Membership.mem e' v)) (And (Membe …
      ⊢ Eq e' e
    -/
  · rw [← Sym2.other_spec h]
    /-
      case h.refine_1
      V : Type u
      G : SimpleGraph V
      v : V
      e : Sym2 V
      he : Membership.mem G.edgeSet e
      h : Membership.mem e v
      e' : Sym2 V
      h' : And (And (Membership.mem G.edgeSet e') (Membership.mem e' v)) (And (Membe …
      ⊢ Eq e' (Sym2.mk { fst := v, snd := Sym2.Mem.other h })
    -/
    exact (Sym2.mem_and_mem_iff (edge_other_ne G he h).symm).mp ⟨h'.1.2, h'.2.2⟩
    /-
      🎉 no goals
    -/
    /-
      case h.refine_2
      V : Type u
      G : SimpleGraph V
      v : V
      e : Sym2 V
      he : Membership.mem G.edgeSet e
      h : Membership.mem e v
      e' : Sym2 V
      ⊢ Eq e' e → And (And (Membership.mem G.edgeSet e') (Membership.mem e' v)) (And …
    -/
  · rintro rfl
    /-
      case h.refine_2
      V : Type u
      G : SimpleGraph V
      v : V
      e' : Sym2 V
      he : Membership.mem G.edgeSet e'
      h : Membership.mem e' v
      ⊢ And (And (Membership.mem G.edgeSet e') (Membership.mem e' v)) (And (Membersh …
    -/
    exact ⟨⟨he, h⟩, he, Sym2.other_mem _⟩
    /-
      🎉 no goals
    -/


theorem compl_neighborSet_disjoint (G : SimpleGraph V) (v : V) :
    Disjoint (G.neighborSet v) (Gᶜ.neighborSet v) := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ Disjoint (G.neighborSet v) ((HasCompl.compl G).neighborSet v)
  -/
  rw [Set.disjoint_iff]
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ HasSubset.Subset (Inter.inter (G.neighborSet v) ((HasCompl.compl G).neighbor …
  -/
  rintro w ⟨h, h'⟩
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    v w : V
    h : Membership.mem (G.neighborSet v) w
    h' : Membership.mem ((HasCompl.compl G).neighborSet v) w
    ⊢ Membership.mem EmptyCollection.emptyCollection w
  -/
  rw [mem_neighborSet, compl_adj] at h'
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    v w : V
    h : Membership.mem (G.neighborSet v) w
    h' : And (Ne v w) (Not (G.Adj v w))
    ⊢ Membership.mem EmptyCollection.emptyCollection w
  -/
  exact h'.2 h
  /-
    🎉 no goals
  -/


theorem neighborSet_union_compl_neighborSet_eq (G : SimpleGraph V) (v : V) :
    G.neighborSet v ∪ Gᶜ.neighborSet v = {v}ᶜ := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ Eq (Union.union (G.neighborSet v) ((HasCompl.compl G).neighborSet v)) (HasCo …
  -/
  ext w
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff (Membership.mem (Union.union (G.neighborSet v) ((HasCompl.compl G).neigh …
  -/
  have h := @ne_of_adj _ G
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    h : ∀ {a b : V}, G.Adj a b → Ne a b
    ⊢ Iff (Membership.mem (Union.union (G.neighborSet v) ((HasCompl.compl G).neigh …
  -/
  simp_rw [Set.mem_union, mem_neighborSet, compl_adj, Set.mem_compl_iff, Set.mem_singleton_iff]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    h : ∀ {a b : V}, G.Adj a b → Ne a b
    ⊢ Iff (Or (G.Adj v w) (And (Ne v w) (Not (G.Adj v w)))) (Not (Eq w v))
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem card_neighborSet_union_compl_neighborSet [Fintype V] (G : SimpleGraph V) (v : V)
    [Fintype (G.neighborSet v ∪ Gᶜ.neighborSet v : Set V)] :
    #(G.neighborSet v ∪ Gᶜ.neighborSet v).toFinset = Fintype.card V - 1 := by
  classical simp_rw [neighborSet_union_compl_neighborSet_eq, Set.toFinset_compl,
      Finset.card_compl, Set.toFinset_card, Set.card_singleton]


theorem neighborSet_compl (G : SimpleGraph V) (v : V) :
    Gᶜ.neighborSet v = (G.neighborSet v)ᶜ \ {v} := by
  /-
    V : Type u
    G : SimpleGraph V
    v : V
    ⊢ Eq ((HasCompl.compl G).neighborSet v) (SDiff.sdiff (HasCompl.compl (G.neighb …
  -/
  ext w
  /-
    case h
    V : Type u
    G : SimpleGraph V
    v w : V
    ⊢ Iff (Membership.mem ((HasCompl.compl G).neighborSet v) w) (Membership.mem (S …
  -/
  simp [and_comm, eq_comm]
  /-
    🎉 no goals
  -/


/-- The set of common neighbors between two vertices `v` and `w` in a graph `G` is the
intersection of the neighbor sets of `v` and `w`. -/
def commonNeighbors (v w : V) : Set V :=
  G.neighborSet v ∩ G.neighborSet w


theorem commonNeighbors_eq (v w : V) : G.commonNeighbors v w = G.neighborSet v ∩ G.neighborSet w :=
  rfl


theorem mem_commonNeighbors {u v w : V} : u ∈ G.commonNeighbors v w ↔ G.Adj v u ∧ G.Adj w u :=
  Iff.rfl


theorem commonNeighbors_symm (v w : V) : G.commonNeighbors v w = G.commonNeighbors w v :=
  Set.inter_comm _ _


theorem not_mem_commonNeighbors_left (v w : V) : v ∉ G.commonNeighbors v w := fun h =>
  ne_of_adj G h.1 rfl


theorem not_mem_commonNeighbors_right (v w : V) : w ∉ G.commonNeighbors v w := fun h =>
  ne_of_adj G h.2 rfl


theorem commonNeighbors_subset_neighborSet_left (v w : V) :
    G.commonNeighbors v w ⊆ G.neighborSet v :=
  Set.inter_subset_left


theorem commonNeighbors_subset_neighborSet_right (v w : V) :
    G.commonNeighbors v w ⊆ G.neighborSet w :=
  Set.inter_subset_right


instance decidableMemCommonNeighbors [DecidableRel G.Adj] (v w : V) :
    DecidablePred (· ∈ G.commonNeighbors v w) :=
  inferInstanceAs <| DecidablePred fun u => u ∈ G.neighborSet v ∧ u ∈ G.neighborSet w


theorem commonNeighbors_top_eq {v w : V} :
    (⊤ : SimpleGraph V).commonNeighbors v w = Set.univ \ {v, w} := by
  /-
    V : Type u
    v w : V
    ⊢ Eq (Top.top.commonNeighbors v w) (SDiff.sdiff Set.univ (Insert.insert v (Sin …
  -/
  ext u
  /-
    case h
    V : Type u
    v w u : V
    ⊢ Iff (Membership.mem (Top.top.commonNeighbors v w) u) (Membership.mem (SDiff. …
  -/
  simp [commonNeighbors, eq_comm, not_or]
  /-
    🎉 no goals
  -/


/-- Given an edge incident to a particular vertex, get the other vertex on the edge. -/
def otherVertexOfIncident {v : V} {e : Sym2 V} (h : e ∈ G.incidenceSet v) : V :=
  Sym2.Mem.other' h.2


theorem edge_other_incident_set {v : V} {e : Sym2 V} (h : e ∈ G.incidenceSet v) :
    e ∈ G.incidenceSet (G.otherVertexOfIncident h) := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    v : V
    e : Sym2 V
    h : Membership.mem (G.incidenceSet v) e
    ⊢ Membership.mem (G.incidenceSet (G.otherVertexOfIncident h)) e
  -/
  use h.1
  /-
    case right
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    v : V
    e : Sym2 V
    h : Membership.mem (G.incidenceSet v) e
    ⊢ Membership.mem e (G.otherVertexOfIncident h)
  -/
  simp [otherVertexOfIncident, Sym2.other_mem']
  /-
    🎉 no goals
  -/


theorem incidence_other_prop {v : V} {e : Sym2 V} (h : e ∈ G.incidenceSet v) :
    G.otherVertexOfIncident h ∈ G.neighborSet v := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    v : V
    e : Sym2 V
    h : Membership.mem (G.incidenceSet v) e
    ⊢ Membership.mem (G.neighborSet v) (G.otherVertexOfIncident h)
  -/
  cases' h with he hv
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    inst✝ : DecidableEq V
    v : V
    e : Sym2 V
    he : Membership.mem G.edgeSet e
    hv : Membership.mem e v
    ⊢ Membership.mem (G.neighborSet v) (G.otherVertexOfIncident ⋯)
  -/
  rwa [← Sym2.other_spec' hv, mem_edgeSet] at he
  /-
    🎉 no goals
  -/

-- Porting note: as a simp lemma this does not apply even to itself

theorem incidence_other_neighbor_edge {v w : V} (h : w ∈ G.neighborSet v) :
    G.otherVertexOfIncident (G.mem_incidence_iff_neighbor.mpr h) = w :=
  Sym2.congr_right.mp (Sym2.other_spec' (G.mem_incidence_iff_neighbor.mpr h).right)


/-- There is an equivalence between the set of edges incident to a given
vertex and the set of vertices adjacent to the vertex. -/
@[simps]
def incidenceSetEquivNeighborSet (v : V) : G.incidenceSet v ≃ G.neighborSet v where
  toFun e := ⟨G.otherVertexOfIncident e.2, G.incidence_other_prop e.2⟩
  invFun w := ⟨s(v, w.1), G.mem_incidence_iff_neighbor.mpr w.2⟩
                   /-
                     ι : Sort u_1
                     V : Type u
                     G : SimpleGraph V
                     a b c u v✝ w : V
                     e : Sym2 V
                     inst✝ : DecidableEq V
                     v : V
                     x : ↑(G.incidenceSet v)
                     ⊢ Eq ((fun w => ⟨Sym2.mk { fst := v, snd := ↑w }, ⋯⟩) ((fun e => ⟨G.otherVerte …
                   -/
  left_inv x := by simp [otherVertexOfIncident]
                   /-
                     🎉 no goals
                   -/
  right_inv := fun ⟨w, hw⟩ => by
    /-
      ι : Sort u_1
      V : Type u
      G : SimpleGraph V
      a b c u v✝ w✝ : V
      e : Sym2 V
      inst✝ : DecidableEq V
      v : V
      x✝ : ↑(G.neighborSet v)
      w : V
      hw : Membership.mem (G.neighborSet v) w
      ⊢ Eq ((fun e => ⟨G.otherVertexOfIncident ⋯, ⋯⟩) ((fun w => ⟨Sym2.mk { fst := v …
    -/
    simp only [mem_neighborSet, Subtype.mk.injEq]
    /-
      ι : Sort u_1
      V : Type u
      G : SimpleGraph V
      a b c u v✝ w✝ : V
      e : Sym2 V
      inst✝ : DecidableEq V
      v : V
      x✝ : ↑(G.neighborSet v)
      w : V
      hw : Membership.mem (G.neighborSet v) w
      ⊢ Eq (G.otherVertexOfIncident ⋯) w
    -/
    exact incidence_other_neighbor_edge _ hw
    /-
      🎉 no goals
    -/


/-- Given a set of vertex pairs, remove all of the corresponding edges from the
graph's edge set, if present.

See also: `SimpleGraph.Subgraph.deleteEdges`. -/
def deleteEdges (s : Set (Sym2 V)) : SimpleGraph V := G \ fromEdgeSet s


@[simp] lemma deleteEdges_adj : (G.deleteEdges s).Adj v w ↔ G.Adj v w ∧ ¬s(v, w) ∈ s :=
  and_congr_right fun h ↦ (and_iff_left h.ne).not


@[simp] lemma deleteEdges_edgeSet (G G' : SimpleGraph V) : G.deleteEdges G'.edgeSet = G \ G' := by
  /-
    V : Type u
    G G' : SimpleGraph V
    ⊢ Eq (G.deleteEdges G'.edgeSet) (SDiff.sdiff G G')
  -/
  ext; simp
       /-
         🎉 no goals
       -/


@[simp]
theorem deleteEdges_deleteEdges (s s' : Set (Sym2 V)) :
                                                                    /-
                                                                      V : Type u
                                                                      G : SimpleGraph V
                                                                      s s' : Set (Sym2 V)
                                                                      ⊢ Eq ((G.deleteEdges s).deleteEdges s') (G.deleteEdges (Union.union s s'))
                                                                    -/
    (G.deleteEdges s).deleteEdges s' = G.deleteEdges (s ∪ s') := by simp [deleteEdges, sdiff_sdiff]
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


                                                            /-
                                                              V : Type u
                                                              G : SimpleGraph V
                                                              ⊢ Eq (G.deleteEdges EmptyCollection.emptyCollection) G
                                                            -/
@[simp] lemma deleteEdges_empty : G.deleteEdges ∅ = G := by simp [deleteEdges]
                                                            /-
                                                              🎉 no goals
                                                            -/

                                                                  /-
                                                                    V : Type u
                                                                    G : SimpleGraph V
                                                                    ⊢ Eq (G.deleteEdges Set.univ) Bot.bot
                                                                  -/
@[simp] lemma deleteEdges_univ : G.deleteEdges Set.univ = ⊥ := by simp [deleteEdges]
                                                                  /-
                                                                    🎉 no goals
                                                                  -/


lemma deleteEdges_le (s : Set (Sym2 V)) : G.deleteEdges s ≤ G := sdiff_le


lemma deleteEdges_anti (h : s₁ ⊆ s₂) : G.deleteEdges s₂ ≤ G.deleteEdges s₁ :=
  sdiff_le_sdiff_left <| fromEdgeSet_mono h


lemma deleteEdges_mono (h : G ≤ H) : G.deleteEdges s ≤ H.deleteEdges s := sdiff_le_sdiff_right h


@[simp] lemma deleteEdges_eq_self : G.deleteEdges s = G ↔ Disjoint G.edgeSet s := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    ⊢ Iff (Eq (G.deleteEdges s) G) (Disjoint G.edgeSet s)
  -/
  rw [deleteEdges, sdiff_eq_left, disjoint_fromEdgeSet]
  /-
    🎉 no goals
  -/


theorem deleteEdges_eq_inter_edgeSet (s : Set (Sym2 V)) :
    G.deleteEdges s = G.deleteEdges (s ∩ G.edgeSet) := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    ⊢ Eq (G.deleteEdges s) (G.deleteEdges (Inter.inter s G.edgeSet))
  -/
  ext
  /-
    case Adj.h.h.a
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    x✝¹ x✝ : V
    ⊢ Iff ((G.deleteEdges s).Adj x✝¹ x✝) ((G.deleteEdges (Inter.inter s G.edgeSet) …
  -/
  simp +contextual [imp_false]
  /-
    🎉 no goals
  -/


theorem deleteEdges_sdiff_eq_of_le {H : SimpleGraph V} (h : H ≤ G) :
    G.deleteEdges (G.edgeSet \ H.edgeSet) = H := by
  /-
    V : Type u
    G H : SimpleGraph V
    h : LE.le H G
    ⊢ Eq (G.deleteEdges (SDiff.sdiff G.edgeSet H.edgeSet)) H
  -/
  rw [← edgeSet_sdiff, deleteEdges_edgeSet, sdiff_sdiff_eq_self h]
  /-
    🎉 no goals
  -/


theorem edgeSet_deleteEdges (s : Set (Sym2 V)) : (G.deleteEdges s).edgeSet = G.edgeSet \ s := by
  /-
    V : Type u
    G : SimpleGraph V
    s : Set (Sym2 V)
    ⊢ Eq (G.deleteEdges s).edgeSet (SDiff.sdiff G.edgeSet s)
  -/
  simp [deleteEdges]
  /-
    🎉 no goals
  -/


