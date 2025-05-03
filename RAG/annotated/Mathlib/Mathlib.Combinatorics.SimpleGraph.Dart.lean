/-- A `Dart` is an oriented edge, implemented as an ordered pair of adjacent vertices.
This terminology comes from combinatorial maps, and they are also known as "half-edges"
or "bonds." -/
structure Dart extends V × V where
  adj : G.Adj fst snd
  deriving DecidableEq


attribute [simp] Dart.adj


theorem Dart.ext_iff (d₁ d₂ : G.Dart) : d₁ = d₂ ↔ d₁.toProd = d₂.toProd := by
  /-
    V : Type u_1
    G : SimpleGraph V
    d₁ d₂ : G.Dart
    ⊢ Iff (Eq d₁ d₂) (Eq d₁.toProd d₂.toProd)
  -/
  cases d₁; cases d₂; simp
                      /-
                        🎉 no goals
                      -/


@[ext]
theorem Dart.ext (d₁ d₂ : G.Dart) (h : d₁.toProd = d₂.toProd) : d₁ = d₂ :=
  (Dart.ext_iff d₁ d₂).mpr h

-- Porting note: deleted `Dart.fst` and `Dart.snd` since they are now invalid declaration names,
-- even though there is not actually a `SimpleGraph.Dart.fst` or `SimpleGraph.Dart.snd`.


@[simp]
theorem Dart.fst_ne_snd (d : G.Dart) : d.fst ≠ d.snd :=
  fun h ↦ G.irrefl (h ▸ d.adj)


@[simp]
theorem Dart.snd_ne_fst (d : G.Dart) : d.snd ≠ d.fst :=
  fun h ↦ G.irrefl (h ▸ d.adj)


theorem Dart.toProd_injective : Function.Injective (Dart.toProd : G.Dart → V × V) :=
  Dart.ext


instance Dart.fintype [Fintype V] [DecidableRel G.Adj] : Fintype G.Dart :=
  Fintype.ofEquiv (Σ v, G.neighborSet v)
    { toFun := fun s => ⟨(s.fst, s.snd), s.snd.property⟩
      invFun := fun d => ⟨d.fst, d.snd, d.adj⟩
                              /-
                                V : Type u_1
                                G : SimpleGraph V
                                inst✝¹ : Fintype V
                                inst✝ : DecidableRel G.Adj
                                s : Sigma fun v => ↑(G.neighborSet v)
                                ⊢ Eq ((fun d => ⟨d.toProd.1, ⟨d.toProd.2, ⋯⟩⟩) ((fun s => { fst := s.fst, snd  …
                              -/
                                      /-
                                        🎉 no goals
                                      -/
      left_inv := fun s => by ext <;> simp
                                      /-
                                        🎉 no goals
                                      -/
                               /-
                                 V : Type u_1
                                 G : SimpleGraph V
                                 inst✝¹ : Fintype V
                                 inst✝ : DecidableRel G.Adj
                                 d : G.Dart
                                 ⊢ Eq ((fun s => { fst := s.fst, snd := ↑s.snd, adj := ⋯ }) ((fun d => ⟨d.toPro …
                               -/
                                       /-
                                         🎉 no goals
                                       -/
      right_inv := fun d => by ext <;> simp }
                                       /-
                                         🎉 no goals
                                       -/


/-- The edge associated to the dart. -/
def Dart.edge (d : G.Dart) : Sym2 V :=
  Sym2.mk d.toProd


@[simp]
theorem Dart.edge_mk {p : V × V} (h : G.Adj p.1 p.2) : (Dart.mk p h).edge = Sym2.mk p :=
  rfl


@[simp]
theorem Dart.edge_mem (d : G.Dart) : d.edge ∈ G.edgeSet :=
  d.adj


/-- The dart with reversed orientation from a given dart. -/
@[simps]
def Dart.symm (d : G.Dart) : G.Dart :=
  ⟨d.toProd.swap, G.symm d.adj⟩


@[simp]
theorem Dart.symm_mk {p : V × V} (h : G.Adj p.1 p.2) : (Dart.mk p h).symm = Dart.mk p.swap h.symm :=
  rfl


@[simp]
theorem Dart.edge_symm (d : G.Dart) : d.symm.edge = d.edge :=
  Sym2.mk_prod_swap_eq


@[simp]
theorem Dart.edge_comp_symm : Dart.edge ∘ Dart.symm = (Dart.edge : G.Dart → Sym2 V) :=
  funext Dart.edge_symm


@[simp]
theorem Dart.symm_symm (d : G.Dart) : d.symm.symm = d :=
  Dart.ext _ _ <| Prod.swap_swap _


@[simp]
theorem Dart.symm_involutive : Function.Involutive (Dart.symm : G.Dart → G.Dart) :=
  Dart.symm_symm


theorem Dart.symm_ne (d : G.Dart) : d.symm ≠ d :=
  ne_of_apply_ne (Prod.snd ∘ Dart.toProd) d.adj.ne


theorem dart_edge_eq_iff : ∀ d₁ d₂ : G.Dart, d₁.edge = d₂.edge ↔ d₁ = d₂ ∨ d₁ = d₂.symm := by
  /-
    V : Type u_1
    G : SimpleGraph V
    ⊢ ∀ (d₁ d₂ : G.Dart), Iff (Eq d₁.edge d₂.edge) (Or (Eq d₁ d₂) (Eq d₁ d₂.symm))
  -/
  rintro ⟨p, hp⟩ ⟨q, hq⟩
  /-
    case mk.mk
    V : Type u_1
    G : SimpleGraph V
    p : Prod V V
    hp : G.Adj p.1 p.2
    q : Prod V V
    hq : G.Adj q.1 q.2
    ⊢ Iff (Eq { toProd := p, adj := hp }.edge { toProd := q, adj := hq }.edge) (Or …
  -/
  simp
  /-
    🎉 no goals
  -/


theorem dart_edge_eq_mk'_iff :
    ∀ {d : G.Dart} {p : V × V}, d.edge = Sym2.mk p ↔ d.toProd = p ∨ d.toProd = p.swap := by
  /-
    V : Type u_1
    G : SimpleGraph V
    ⊢ ∀ {d : G.Dart} {p : Prod V V}, Iff (Eq d.edge (Sym2.mk p)) (Or (Eq d.toProd  …
  -/
  rintro ⟨p, h⟩
  /-
    case mk
    V : Type u_1
    G : SimpleGraph V
    p : Prod V V
    h : G.Adj p.1 p.2
    ⊢ ∀ {p_1 : Prod V V}, Iff (Eq { toProd := p, adj := h }.edge (Sym2.mk p_1)) (O …
  -/
  apply Sym2.mk_eq_mk_iff
  /-
    🎉 no goals
  -/


theorem dart_edge_eq_mk'_iff' :
    ∀ {d : G.Dart} {u v : V},
      d.edge = s(u, v) ↔ d.fst = u ∧ d.snd = v ∨ d.fst = v ∧ d.snd = u := by
  /-
    V : Type u_1
    G : SimpleGraph V
    ⊢ ∀ {d : G.Dart} {u v : V}, Iff (Eq d.edge (Sym2.mk { fst := u, snd := v })) ( …
  -/
  rintro ⟨⟨a, b⟩, h⟩ u v
  /-
    case mk.mk
    V : Type u_1
    G : SimpleGraph V
    a b : V
    h : G.Adj { fst := a, snd := b }.1 { fst := a, snd := b }.2
    u v : V
    ⊢ Iff (Eq { fst := a, snd := b, adj := h }.edge (Sym2.mk { fst := u, snd := v  …
  -/
  rw [dart_edge_eq_mk'_iff]
  /-
    case mk.mk
    V : Type u_1
    G : SimpleGraph V
    a b : V
    h : G.Adj { fst := a, snd := b }.1 { fst := a, snd := b }.2
    u v : V
    ⊢ Iff (Or (Eq { fst := a, snd := b, adj := h }.toProd { fst := u, snd := v })  …
  -/
  simp
  /-
    🎉 no goals
  -/


/-- Two darts are said to be adjacent if they could be consecutive
darts in a walk -- that is, the first dart's second vertex is equal to
the second dart's first vertex. -/
def DartAdj (d d' : G.Dart) : Prop :=
  d.snd = d'.fst


/-- For a given vertex `v`, this is the bijective map from the neighbor set at `v`
to the darts `d` with `d.fst = v`. -/
@[simps]
def dartOfNeighborSet (v : V) (w : G.neighborSet v) : G.Dart :=
  ⟨(v, w), w.property⟩


theorem dartOfNeighborSet_injective (v : V) : Function.Injective (G.dartOfNeighborSet v) :=
  fun e₁ e₂ h =>
  Subtype.ext <| by
    /-
      V : Type u_1
      G : SimpleGraph V
      v : V
      e₁ e₂ : ↑(G.neighborSet v)
      h : Eq (G.dartOfNeighborSet v e₁) (G.dartOfNeighborSet v e₂)
      ⊢ Eq ↑e₁ ↑e₂
    -/
    injection h with h'
    /-
      V : Type u_1
      G : SimpleGraph V
      v : V
      e₁ e₂ : ↑(G.neighborSet v)
      h' : Eq { fst := v, snd := ↑e₁ } { fst := v, snd := ↑e₂ }
      ⊢ Eq ↑e₁ ↑e₂
    -/
    convert congr_arg Prod.snd h'
    /-
      🎉 no goals
    -/


instance nonempty_dart_top [Nontrivial V] : Nonempty (⊤ : SimpleGraph V).Dart := by
  /-
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Nontrivial V
    ⊢ Nonempty Top.top.Dart
  -/
  obtain ⟨v, w, h⟩ := exists_pair_ne V
  /-
    case intro.intro
    V : Type u_1
    G : SimpleGraph V
    inst✝ : Nontrivial V
    v w : V
    h : Ne v w
    ⊢ Nonempty Top.top.Dart
  -/
  exact ⟨⟨(v, w), h⟩⟩
  /-
    🎉 no goals
  -/


