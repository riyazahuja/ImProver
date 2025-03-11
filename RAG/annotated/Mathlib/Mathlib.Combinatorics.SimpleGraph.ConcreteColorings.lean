theorem two_le_chromaticNumber_of_adj {α} {G : SimpleGraph α} {u v : α} (hAdj : G.Adj u v) :
    2 ≤ G.chromaticNumber := by
  /-
    α : Type u_1
    G : SimpleGraph α
    u v : α
    hAdj : G.Adj u v
    ⊢ LE.le 2 G.chromaticNumber
  -/
  refine le_of_not_lt ?_
  /-
    α : Type u_1
    G : SimpleGraph α
    u v : α
    hAdj : G.Adj u v
    ⊢ Not (LT.lt G.chromaticNumber 2)
  -/
  intro h
  /-
    α : Type u_1
    G : SimpleGraph α
    u v : α
    hAdj : G.Adj u v
    h : LT.lt G.chromaticNumber 2
    ⊢ False
  -/
  have hc : G.Colorable 1 := chromaticNumber_le_iff_colorable.mp (Order.le_of_lt_add_one h)
  /-
    α : Type u_1
    G : SimpleGraph α
    u v : α
    hAdj : G.Adj u v
    h : LT.lt G.chromaticNumber 2
    hc : G.Colorable 1
    ⊢ False
  -/
  let c : G.Coloring (Fin 1) := hc.some
  /-
    α : Type u_1
    G : SimpleGraph α
    u v : α
    hAdj : G.Adj u v
    h : LT.lt G.chromaticNumber 2
    hc : G.Colorable 1
    c : G.Coloring (Fin 1) := Nonempty.some hc
    ⊢ False
  -/
  exact c.valid hAdj (Subsingleton.elim (c u) (c v))
  /-
    🎉 no goals
  -/


/-- Bicoloring of a path graph -/
def pathGraph.bicoloring (n : ℕ) :
    Coloring (pathGraph n) Bool :=
  Coloring.mk (fun u ↦ u.val % 2 = 0) <| by
    /-
      n : Nat
      ⊢ ∀ {v w : Fin n}, (SimpleGraph.pathGraph n).Adj v w → Ne ((fun u => Decidable …
    -/
    intro u v
    /-
      n : Nat
      u v : Fin n
      ⊢ (SimpleGraph.pathGraph n).Adj u v → Ne ((fun u => Decidable.decide (Eq (HMod …
    -/
    rw [pathGraph_adj]
    /-
      n : Nat
      u v : Fin n
      ⊢ Or (Eq (HAdd.hAdd (↑u) 1) ↑v) (Eq (HAdd.hAdd (↑v) 1) ↑u) → Ne ((fun u => Dec …
    -/
                       /-
                         🎉 no goals
                       -/
    rintro (h | h) <;> simp [← h, not_iff, Nat.succ_mod_two_eq_zero_iff]
                       /-
                         🎉 no goals
                       -/


/-- Embedding of `pathGraph 2` into the first two elements of `pathGraph n` for `2 ≤ n` -/
def pathGraph_two_embedding (n : ℕ) (h : 2 ≤ n) : pathGraph 2 ↪g pathGraph n where
  toFun v := ⟨v, trans v.2 h⟩
  inj' := by
    /-
      n : Nat
      h : LE.le 2 n
      ⊢ Function.Injective fun v => ⟨↑v, ⋯⟩
    -/
    rintro v w
    /-
      n : Nat
      h : LE.le 2 n
      v w : Fin 2
      ⊢ Eq ((fun v => ⟨↑v, ⋯⟩) v) ((fun v => ⟨↑v, ⋯⟩) w) → Eq v w
    -/
    rw [Fin.mk.injEq]
    /-
      n : Nat
      h : LE.le 2 n
      v w : Fin 2
      ⊢ Eq ↑v ↑w → Eq v w
    -/
    exact Fin.ext
    /-
      🎉 no goals
    -/
  map_rel_iff' := by
    /-
      n : Nat
      h : LE.le 2 n
      ⊢ ∀ {a b : Fin 2}, Iff ((SimpleGraph.pathGraph n).Adj ({ toFun := fun v => ⟨↑v …
    -/
    intro v w
    /-
      n : Nat
      h : LE.le 2 n
      v w : Fin 2
      ⊢ Iff ((SimpleGraph.pathGraph n).Adj ({ toFun := fun v => ⟨↑v, ⋯⟩, inj' := ⋯ } …
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
    fin_cases v <;> fin_cases w <;> simp [pathGraph, ← Fin.coe_covBy_iff]
                                    /-
                                      🎉 no goals
                                    -/


theorem chromaticNumber_pathGraph (n : ℕ) (h : 2 ≤ n) :
    (pathGraph n).chromaticNumber = 2 := by
  /-
    n : Nat
    h : LE.le 2 n
    ⊢ Eq (SimpleGraph.pathGraph n).chromaticNumber 2
  -/
  have hc := (pathGraph.bicoloring n).colorable
  /-
    n : Nat
    h : LE.le 2 n
    hc : (SimpleGraph.pathGraph n).Colorable (Fintype.card Bool)
    ⊢ Eq (SimpleGraph.pathGraph n).chromaticNumber 2
  -/
  apply le_antisymm
    /-
      case a
      n : Nat
      h : LE.le 2 n
      hc : (SimpleGraph.pathGraph n).Colorable (Fintype.card Bool)
      ⊢ LE.le (SimpleGraph.pathGraph n).chromaticNumber 2
    -/
  · exact hc.chromaticNumber_le
    /-
      🎉 no goals
    -/
    /-
      case a
      n : Nat
      h : LE.le 2 n
      hc : (SimpleGraph.pathGraph n).Colorable (Fintype.card Bool)
      ⊢ LE.le 2 (SimpleGraph.pathGraph n).chromaticNumber
    -/
  · have hAdj : (pathGraph n).Adj ⟨0, Nat.zero_lt_of_lt h⟩ ⟨1, h⟩ := by simp [pathGraph_adj]
    /-
      case a
      n : Nat
      h : LE.le 2 n
      hc : (SimpleGraph.pathGraph n).Colorable (Fintype.card Bool)
      hAdj : (SimpleGraph.pathGraph n).Adj ⟨0, ⋯⟩ ⟨1, h⟩
      ⊢ LE.le 2 (SimpleGraph.pathGraph n).chromaticNumber
    -/
    exact two_le_chromaticNumber_of_adj hAdj
    /-
      🎉 no goals
    -/


theorem Coloring.even_length_iff_congr {α} {G : SimpleGraph α}
    (c : G.Coloring Bool) {u v : α} (p : G.Walk u v) :
    Even p.length ↔ (c u ↔ c v) := by
  induction p with
  | nil => simp
  | @cons u v w h p ih =>
    simp only [Walk.length_cons, Nat.even_add_one]
    have : ¬ c u = true ↔ c v = true := by
      rw [← not_iff, ← Bool.eq_iff_iff]
      exact c.valid h
    tauto


theorem Coloring.odd_length_iff_not_congr {α} {G : SimpleGraph α}
    (c : G.Coloring Bool) {u v : α} (p : G.Walk u v) :
    Odd p.length ↔ (¬c u ↔ c v) := by
  /-
    α : Type u_1
    G : SimpleGraph α
    c : G.Coloring Bool
    u v : α
    p : G.Walk u v
    ⊢ Iff (Odd p.length) (Iff (Not (Eq (c u) Bool.true)) (Eq (c v) Bool.true))
  -/
  rw [← Nat.not_even_iff_odd, c.even_length_iff_congr p]
  /-
    α : Type u_1
    G : SimpleGraph α
    c : G.Coloring Bool
    u v : α
    p : G.Walk u v
    ⊢ Iff (Not (Iff (Eq (c u) Bool.true) (Eq (c v) Bool.true))) (Iff (Not (Eq (c u …
  -/
  tauto
  /-
    🎉 no goals
  -/


theorem Walk.three_le_chromaticNumber_of_odd_loop {α} {G : SimpleGraph α} {u : α} (p : G.Walk u u)
    (hOdd : Odd p.length) : 3 ≤ G.chromaticNumber := Classical.by_contradiction <| by
  /-
    α : Type u_1
    G : SimpleGraph α
    u : α
    p : G.Walk u u
    hOdd : Odd p.length
    ⊢ Not (LE.le 3 G.chromaticNumber) → False
  -/
  intro h
  /-
    α : Type u_1
    G : SimpleGraph α
    u : α
    p : G.Walk u u
    hOdd : Odd p.length
    h : Not (LE.le 3 G.chromaticNumber)
    ⊢ False
  -/
  have h' : G.chromaticNumber ≤ 2 := Order.le_of_lt_add_one <| not_le.mp h
  /-
    α : Type u_1
    G : SimpleGraph α
    u : α
    p : G.Walk u u
    hOdd : Odd p.length
    h : Not (LE.le 3 G.chromaticNumber)
    h' : LE.le G.chromaticNumber 2
    ⊢ False
  -/
  let c : G.Coloring (Fin 2) := (chromaticNumber_le_iff_colorable.mp h').some
  /-
    α : Type u_1
    G : SimpleGraph α
    u : α
    p : G.Walk u u
    hOdd : Odd p.length
    h : Not (LE.le 3 G.chromaticNumber)
    h' : LE.le G.chromaticNumber 2
    c : G.Coloring (Fin 2) := Nonempty.some ⋯
    ⊢ False
  -/
  let c' : G.Coloring Bool := recolorOfEquiv G finTwoEquiv c
  /-
    α : Type u_1
    G : SimpleGraph α
    u : α
    p : G.Walk u u
    hOdd : Odd p.length
    h : Not (LE.le 3 G.chromaticNumber)
    h' : LE.le G.chromaticNumber 2
    c : G.Coloring (Fin 2) := Nonempty.some ⋯
    c' : G.Coloring Bool := (G.recolorOfEquiv finTwoEquiv) c
    ⊢ False
  -/
  have : ¬c' u ↔ c' u := (c'.odd_length_iff_not_congr p).mp hOdd
  /-
    α : Type u_1
    G : SimpleGraph α
    u : α
    p : G.Walk u u
    hOdd : Odd p.length
    h : Not (LE.le 3 G.chromaticNumber)
    h' : LE.le G.chromaticNumber 2
    c : G.Coloring (Fin 2) := Nonempty.some ⋯
    c' : G.Coloring Bool := (G.recolorOfEquiv finTwoEquiv) c
    this : Iff (Not (Eq (c' u) Bool.true)) (Eq (c' u) Bool.true)
    ⊢ False
  -/
  simp_all
  /-
    🎉 no goals
  -/


