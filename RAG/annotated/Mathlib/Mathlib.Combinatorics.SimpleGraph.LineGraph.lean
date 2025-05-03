/--
The line graph of a simple graph `G` has its vertex set as the edges of `G`, and two vertices of
the line graph are adjacent if the corresponding edges share a vertex in `G`.
-/
def lineGraph {V : Type*} (G : SimpleGraph V) : SimpleGraph G.edgeSet where
  Adj e₁ e₂ := e₁ ≠ e₂ ∧ (e₁ ∩ e₂ : Set V).Nonempty
                   /-
                     V✝ : Type u_1
                     G✝ : SimpleGraph V✝
                     V : Type u_2
                     G : SimpleGraph V
                     e₁ e₂ : ↑G.edgeSet
                     ⊢ (fun e₁ e₂ => And (Ne e₁ e₂) (Inter.inter ↑↑e₁ ↑↑e₂).Nonempty) e₁ e₂ → (fun  …
                   -/
  symm e₁ e₂ := by intro h; rwa [ne_comm, Set.inter_comm]
                            /-
                              🎉 no goals
                            -/


lemma lineGraph_adj_iff_exists {e₁ e₂ : G.edgeSet} :
    (G.lineGraph).Adj e₁ e₂ ↔ e₁ ≠ e₂ ∧ ∃ v, v ∈ (e₁ : Sym2 V) ∧ v ∈ (e₂ : Sym2 V) := by
  /-
    V : Type u_1
    G : SimpleGraph V
    e₁ e₂ : ↑G.edgeSet
    ⊢ Iff (G.lineGraph.Adj e₁ e₂) (And (Ne e₁ e₂) (Exists fun v => And (Membership …
  -/
  simp [Set.Nonempty, lineGraph]
  /-
    🎉 no goals
  -/


                                                                      /-
                                                                        V : Type u_1
                                                                        ⊢ Eq Bot.bot.lineGraph Bot.bot
                                                                      -/
@[simp] lemma lineGraph_bot : (⊥ : SimpleGraph V).lineGraph = ⊥ := by aesop (add simp lineGraph)
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


