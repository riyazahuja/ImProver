theorem dart_fst_fiber [DecidableEq V] (v : V) :
    ({d : G.Dart | d.fst = v} : Finset _) = univ.image (G.dartOfNeighborSet v) := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v : V
    ⊢ Eq (Finset.filter (fun d => Eq d.toProd.1 v) Finset.univ) (Finset.image (G.d …
  -/
  ext d
  /-
    case h
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v : V
    d : G.Dart
    ⊢ Iff (Membership.mem (Finset.filter (fun d => Eq d.toProd.1 v) Finset.univ) d …
  -/
  simp only [mem_image, true_and, mem_filter, SetCoe.exists, mem_univ, exists_prop_of_true]
  /-
    case h
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v : V
    d : G.Dart
    ⊢ Iff (Eq d.toProd.1 v) (Exists fun x => Exists fun h => Eq (G.dartOfNeighborS …
  -/
  constructor
    /-
      case h.mp
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq V
      v : V
      d : G.Dart
      ⊢ Eq d.toProd.1 v → Exists fun x => Exists fun h => Eq (G.dartOfNeighborSet v  …
    -/
  · rintro rfl
    /-
      case h.mp
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq V
      d : G.Dart
      ⊢ Exists fun x => Exists fun h => Eq (G.dartOfNeighborSet d.toProd.1 ⟨x, h⟩) d
    -/
    exact ⟨_, d.adj, by ext <;> rfl⟩
    /-
      🎉 no goals
    -/
    /-
      case h.mpr
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq V
      v : V
      d : G.Dart
      ⊢ (Exists fun x => Exists fun h => Eq (G.dartOfNeighborSet v ⟨x, h⟩) d) → Eq d …
    -/
  · rintro ⟨e, he, rfl⟩
    /-
      case h.mpr.intro.intro
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableRel G.Adj
      inst✝ : DecidableEq V
      v e : V
      he : Membership.mem (G.neighborSet v) e
      ⊢ Eq (G.dartOfNeighborSet v ⟨e, he⟩).toProd.1 v
    -/
    rfl
    /-
      🎉 no goals
    -/


theorem dart_fst_fiber_card_eq_degree [DecidableEq V] (v : V) :
    #{d : G.Dart | d.fst = v} = G.degree v := by
  simpa only [dart_fst_fiber, Finset.card_univ, card_neighborSet_eq_degree] using
    card_image_of_injective univ (G.dartOfNeighborSet_injective v)


theorem dart_card_eq_sum_degrees : Fintype.card G.Dart = ∑ v, G.degree v := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Eq (Fintype.card G.Dart) (Finset.univ.sum fun v => G.degree v)
  -/
  haveI := Classical.decEq V
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    this : DecidableEq V
    ⊢ Eq (Fintype.card G.Dart) (Finset.univ.sum fun v => G.degree v)
  -/
  simp only [← card_univ, ← dart_fst_fiber_card_eq_degree]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    this : DecidableEq V
    ⊢ Eq Finset.univ.card (Finset.univ.sum fun x => (Finset.filter (fun d => Eq d. …
  -/
  exact card_eq_sum_card_fiberwise (by simp)
  /-
    🎉 no goals
  -/


theorem Dart.edge_fiber [DecidableEq V] (d : G.Dart) :
    ({d' : G.Dart | d'.edge = d.edge} : Finset _) = {d, d.symm} :=
                          /-
                            V : Type u
                            G : SimpleGraph V
                            inst✝² : Fintype V
                            inst✝¹ : DecidableRel G.Adj
                            inst✝ : DecidableEq V
                            d d' : G.Dart
                            ⊢ Iff (Membership.mem (Finset.filter (fun d' => Eq d'.edge d.edge) Finset.univ …
                          -/
  Finset.ext fun d' => by simpa using dart_edge_eq_iff d' d
                          /-
                            🎉 no goals
                          -/


theorem dart_edge_fiber_card [DecidableEq V] (e : Sym2 V) (h : e ∈ G.edgeSet) :
    #{d : G.Dart | d.edge = e} = 2 := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    e : Sym2 V
    h : Membership.mem G.edgeSet e
    ⊢ Eq (Finset.filter (fun d => Eq d.edge e) Finset.univ).card 2
  -/
  induction' e with v w
  /-
    case h
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w : V
    h : Membership.mem G.edgeSet (Sym2.mk { fst := v, snd := w })
    ⊢ Eq (Finset.filter (fun d => Eq d.edge (Sym2.mk { fst := v, snd := w })) Fins …
  -/
  let d : G.Dart := ⟨(v, w), h⟩
  /-
    case h
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w : V
    h : Membership.mem G.edgeSet (Sym2.mk { fst := v, snd := w })
    d : G.Dart := { fst := v, snd := w, adj := h }
    ⊢ Eq (Finset.filter (fun d => Eq d.edge (Sym2.mk { fst := v, snd := w })) Fins …
  -/
  convert congr_arg card d.edge_fiber
  /-
    case h.e'_3
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w : V
    h : Membership.mem G.edgeSet (Sym2.mk { fst := v, snd := w })
    d : G.Dart := { fst := v, snd := w, adj := h }
    ⊢ Eq 2 (Insert.insert d (Singleton.singleton d.symm)).card
  -/
  rw [card_insert_of_not_mem, card_singleton]
  /-
    case h.e'_3
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w : V
    h : Membership.mem G.edgeSet (Sym2.mk { fst := v, snd := w })
    d : G.Dart := { fst := v, snd := w, adj := h }
    ⊢ Not (Membership.mem (Singleton.singleton d.symm) d)
  -/
  rw [mem_singleton]
  /-
    case h.e'_3
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableRel G.Adj
    inst✝ : DecidableEq V
    v w : V
    h : Membership.mem G.edgeSet (Sym2.mk { fst := v, snd := w })
    d : G.Dart := { fst := v, snd := w, adj := h }
    ⊢ Not (Eq d d.symm)
  -/
  exact d.symm_ne.symm
  /-
    🎉 no goals
  -/


theorem dart_card_eq_twice_card_edges : Fintype.card G.Dart = 2 * #G.edgeFinset := by
  classical
  rw [← card_univ]
  rw [@card_eq_sum_card_fiberwise _ _ _ Dart.edge _ G.edgeFinset fun d _h =>
      by rw [mem_edgeFinset]; apply Dart.edge_mem]
  rw [← mul_comm, sum_const_nat]
  intro e h
  apply G.dart_edge_fiber_card e
  rwa [← mem_edgeFinset]


/-- The degree-sum formula.  This is also known as the handshaking lemma, which might
more specifically refer to `SimpleGraph.even_card_odd_degree_vertices`. -/
theorem sum_degrees_eq_twice_card_edges : ∑ v, G.degree v = 2 * #G.edgeFinset :=
  G.dart_card_eq_sum_degrees.symm.trans G.dart_card_eq_twice_card_edges


lemma two_mul_card_edgeFinset : 2 * #G.edgeFinset = #(univ.filter fun (x, y) ↦ G.Adj x y) := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Eq (HMul.hMul 2 G.edgeFinset.card) (Finset.filter (fun x => SimpleGraph.two_ …
  -/
  rw [← dart_card_eq_twice_card_edges, ← card_univ]
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    ⊢ Eq Finset.univ.card (Finset.filter (fun x => SimpleGraph.two_mul_card_edgeFi …
  -/
  refine card_bij' (fun d _ ↦ (d.fst, d.snd)) (fun xy h ↦ ⟨xy, (mem_filter.1 h).2⟩) ?_ ?_ ?_ ?_
        /-
          case refine_1
          V : Type u
          G : SimpleGraph V
          inst✝¹ : Fintype V
          inst✝ : DecidableRel G.Adj
          ⊢ ∀ (a : G.Dart) (ha : Membership.mem Finset.univ a), Membership.mem (Finset.f …
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
    <;> simp
        /-
          🎉 no goals
        -/


/-- The handshaking lemma.  See also `SimpleGraph.sum_degrees_eq_twice_card_edges`. -/
theorem even_card_odd_degree_vertices [Fintype V] [DecidableRel G.Adj] :
    Even #{v | Odd (G.degree v)} := by
  classical
    have h := congr_arg (fun n => ↑n : ℕ → ZMod 2) G.sum_degrees_eq_twice_card_edges
    simp only [ZMod.natCast_self, zero_mul, Nat.cast_mul] at h
    rw [Nat.cast_sum, ← sum_filter_ne_zero] at h
    rw [@sum_congr _ _ _ _ (fun v => (G.degree v : ZMod 2)) (fun _v => (1 : ZMod 2)) _ rfl] at h
    · simp only [filter_congr, mul_one, nsmul_eq_mul, sum_const, Ne] at h
      rw [← ZMod.eq_zero_iff_even]
      convert h
      exact ZMod.ne_zero_iff_odd.symm
    · intro v
      simp only [true_and, mem_filter, mem_univ, Ne]
      rw [ZMod.eq_zero_iff_even, ZMod.eq_one_iff_odd, ← Nat.not_even_iff_odd, imp_self]
      trivial


theorem odd_card_odd_degree_vertices_ne [Fintype V] [DecidableEq V] [DecidableRel G.Adj] (v : V)
    (h : Odd (G.degree v)) : Odd #{w | w ≠ v ∧ Odd (G.degree w)} := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    inst✝ : DecidableRel G.Adj
    v : V
    h : Odd (G.degree v)
    ⊢ Odd (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w))) Finset.univ).c …
  -/
  rcases G.even_card_odd_degree_vertices with ⟨k, hg⟩
  have hk : 0 < k := by
    have hh : Finset.Nonempty {v : V | Odd (G.degree v)} := by
      use v
      simp only [true_and, mem_filter, mem_univ]
      exact h
    rwa [← card_pos, hg, ← two_mul, mul_pos_iff_of_pos_left] at hh
    exact zero_lt_two
  have hc : (fun w : V => w ≠ v ∧ Odd (G.degree w)) = fun w : V => Odd (G.degree w) ∧ w ≠ v := by
    ext w
    rw [and_comm]
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    inst✝ : DecidableRel G.Adj
    v : V
    h : Odd (G.degree v)
    k : Nat
    hg : Eq (Finset.filter (fun v => Odd (G.degree v)) Finset.univ).card (HAdd.hAd …
    hk : LT.lt 0 k
    hc : Eq (fun w => And (Ne w v) (Odd (G.degree w))) fun w => And (Odd (G.degree …
    ⊢ Odd (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w))) Finset.univ).c …
  -/
  simp only [hc, filter_congr]
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    inst✝² : Fintype V
    inst✝¹ : DecidableEq V
    inst✝ : DecidableRel G.Adj
    v : V
    h : Odd (G.degree v)
    k : Nat
    hg : Eq (Finset.filter (fun v => Odd (G.degree v)) Finset.univ).card (HAdd.hAd …
    hk : LT.lt 0 k
    hc : Eq (fun w => And (Ne w v) (Odd (G.degree w))) fun w => And (Odd (G.degree …
    ⊢ Odd (Finset.filter (fun w => And (Odd (G.degree w)) (Ne w v)) Finset.univ).c …
  -/
  rw [← filter_filter, filter_ne', card_erase_of_mem]
    /-
      case intro
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableEq V
      inst✝ : DecidableRel G.Adj
      v : V
      h : Odd (G.degree v)
      k : Nat
      hg : Eq (Finset.filter (fun v => Odd (G.degree v)) Finset.univ).card (HAdd.hAd …
      hk : LT.lt 0 k
      hc : Eq (fun w => And (Ne w v) (Odd (G.degree w))) fun w => And (Odd (G.degree …
      ⊢ Odd (HSub.hSub (Finset.filter (fun w => Odd (G.degree w)) Finset.univ).card 1)
    -/
  · refine ⟨k - 1, tsub_eq_of_eq_add <| hg.trans ?_⟩
    /-
      case intro
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableEq V
      inst✝ : DecidableRel G.Adj
      v : V
      h : Odd (G.degree v)
      k : Nat
      hg : Eq (Finset.filter (fun v => Odd (G.degree v)) Finset.univ).card (HAdd.hAd …
      hk : LT.lt 0 k
      hc : Eq (fun w => And (Ne w v) (Odd (G.degree w))) fun w => And (Odd (G.degree …
      ⊢ Eq (HAdd.hAdd k k) (HAdd.hAdd (HAdd.hAdd (HMul.hMul 2 (HSub.hSub k 1)) 1) 1)
    -/
    rw [add_assoc, one_add_one_eq_two, ← Nat.mul_succ, ← two_mul]
    /-
      case intro
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableEq V
      inst✝ : DecidableRel G.Adj
      v : V
      h : Odd (G.degree v)
      k : Nat
      hg : Eq (Finset.filter (fun v => Odd (G.degree v)) Finset.univ).card (HAdd.hAd …
      hk : LT.lt 0 k
      hc : Eq (fun w => And (Ne w v) (Odd (G.degree w))) fun w => And (Odd (G.degree …
      ⊢ Eq (HMul.hMul 2 k) (HMul.hMul 2 (HSub.hSub k 1).succ)
    -/
    congr
    /-
      case intro.e_a
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableEq V
      inst✝ : DecidableRel G.Adj
      v : V
      h : Odd (G.degree v)
      k : Nat
      hg : Eq (Finset.filter (fun v => Odd (G.degree v)) Finset.univ).card (HAdd.hAd …
      hk : LT.lt 0 k
      hc : Eq (fun w => And (Ne w v) (Odd (G.degree w))) fun w => And (Odd (G.degree …
      ⊢ Eq k (HSub.hSub k 1).succ
    -/
    omega
    /-
      🎉 no goals
    -/
    /-
      case intro
      V : Type u
      G : SimpleGraph V
      inst✝² : Fintype V
      inst✝¹ : DecidableEq V
      inst✝ : DecidableRel G.Adj
      v : V
      h : Odd (G.degree v)
      k : Nat
      hg : Eq (Finset.filter (fun v => Odd (G.degree v)) Finset.univ).card (HAdd.hAd …
      hk : LT.lt 0 k
      hc : Eq (fun w => And (Ne w v) (Odd (G.degree w))) fun w => And (Odd (G.degree …
      ⊢ Membership.mem (Finset.filter (fun w => Odd (G.degree w)) Finset.univ) v
    -/
  · simpa only [true_and, mem_filter, mem_univ]
    /-
      🎉 no goals
    -/


theorem exists_ne_odd_degree_of_exists_odd_degree [Fintype V] [DecidableRel G.Adj] (v : V)
    (h : Odd (G.degree v)) : ∃ w : V, w ≠ v ∧ Odd (G.degree w) := by
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    h : Odd (G.degree v)
    ⊢ Exists fun w => And (Ne w v) (Odd (G.degree w))
  -/
  haveI := Classical.decEq V
  /-
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    h : Odd (G.degree v)
    this : DecidableEq V
    ⊢ Exists fun w => And (Ne w v) (Odd (G.degree w))
  -/
  rcases G.odd_card_odd_degree_vertices_ne v h with ⟨k, hg⟩
  have hg' : 0 < #{w | w ≠ v ∧ Odd (G.degree w)} := by
    rw [hg]
    apply Nat.succ_pos
  /-
    case intro
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    h : Odd (G.degree v)
    this : DecidableEq V
    k : Nat
    hg : Eq (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w))) Finset.univ) …
    hg' : LT.lt 0 (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w))) Finset …
    ⊢ Exists fun w => And (Ne w v) (Odd (G.degree w))
  -/
  rcases card_pos.mp hg' with ⟨w, hw⟩
  /-
    case intro.intro
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    h : Odd (G.degree v)
    this : DecidableEq V
    k : Nat
    hg : Eq (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w))) Finset.univ) …
    hg' : LT.lt 0 (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w))) Finset …
    w : V
    hw : Membership.mem (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w)))  …
    ⊢ Exists fun w => And (Ne w v) (Odd (G.degree w))
  -/
  simp only [true_and, mem_filter, mem_univ, Ne] at hw
  /-
    case intro.intro
    V : Type u
    G : SimpleGraph V
    inst✝¹ : Fintype V
    inst✝ : DecidableRel G.Adj
    v : V
    h : Odd (G.degree v)
    this : DecidableEq V
    k : Nat
    hg : Eq (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w))) Finset.univ) …
    hg' : LT.lt 0 (Finset.filter (fun w => And (Ne w v) (Odd (G.degree w))) Finset …
    w : V
    hw : And (Not (Eq w v)) (Odd (G.degree w))
    ⊢ Exists fun w => And (Ne w v) (Odd (G.degree w))
  -/
  exact ⟨w, hw⟩
  /-
    🎉 no goals
  -/


