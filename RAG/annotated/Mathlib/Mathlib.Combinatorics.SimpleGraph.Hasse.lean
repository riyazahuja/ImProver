/-- The Hasse diagram of an order as a simple graph. The graph of the covering relation. -/
def hasse : SimpleGraph α where
  Adj a b := a ⋖ b ∨ b ⋖ a
  symm _a _b := Or.symm
  loopless _a h := h.elim (irrefl _) (irrefl _)


@[simp]
theorem hasse_adj : (hasse α).Adj a b ↔ a ⋖ b ∨ b ⋖ a :=
  Iff.rfl


/-- `αᵒᵈ` and `α` have the same Hasse diagram. -/
def hasseDualIso : hasse αᵒᵈ ≃g hasse α :=
                                   /-
                                     α : Type u_1
                                     β : Type u_2
                                     inst✝ : Preorder α
                                     a b : α
                                     ⊢ ∀ {a b : OrderDual α}, Iff ((SimpleGraph.hasse α).Adj (__src✝ a) (__src✝ b)) …
                                   -/
  { ofDual with map_rel_iff' := by simp [or_comm] }
                                   /-
                                     🎉 no goals
                                   -/


@[simp]
theorem hasseDualIso_apply (a : αᵒᵈ) : hasseDualIso a = ofDual a :=
  rfl


@[simp]
theorem hasseDualIso_symm_apply (a : α) : hasseDualIso.symm a = toDual a :=
  rfl


@[simp]
theorem hasse_prod : hasse (α × β) = hasse α □ hasse β := by
  /-
    α : Type u_1
    β : Type u_2
    inst✝¹ : PartialOrder α
    inst✝ : PartialOrder β
    ⊢ Eq (SimpleGraph.hasse (Prod α β)) ((SimpleGraph.hasse α).boxProd (SimpleGrap …
  -/
  ext x y
  simp_rw [boxProd_adj, hasse_adj, Prod.covBy_iff, or_and_right, @eq_comm _ y.1, @eq_comm _ y.2,
    or_or_or_comm]


theorem hasse_preconnected_of_succ [SuccOrder α] [IsSuccArchimedean α] : (hasse α).Preconnected :=
  fun a b => by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : SuccOrder α
    inst✝ : IsSuccArchimedean α
    a b : α
    ⊢ (SimpleGraph.hasse α).Reachable a b
  -/
  rw [reachable_iff_reflTransGen]
  exact
    reflTransGen_of_succ _ (fun c hc => Or.inl <| covBy_succ_of_not_isMax hc.2.not_isMax)
      fun c hc => Or.inr <| covBy_succ_of_not_isMax hc.2.not_isMax


theorem hasse_preconnected_of_pred [PredOrder α] [IsPredArchimedean α] : (hasse α).Preconnected :=
  fun a b => by
  /-
    α : Type u_1
    inst✝² : LinearOrder α
    inst✝¹ : PredOrder α
    inst✝ : IsPredArchimedean α
    a b : α
    ⊢ (SimpleGraph.hasse α).Reachable a b
  -/
  rw [reachable_iff_reflTransGen, ← reflTransGen_swap]
  exact
    reflTransGen_of_pred _ (fun c hc => Or.inl <| pred_covBy_of_not_isMin hc.1.not_isMin)
      fun c hc => Or.inr <| pred_covBy_of_not_isMin hc.1.not_isMin


/-- The path graph on `n` vertices. -/
def pathGraph (n : ℕ) : SimpleGraph (Fin n) :=
  hasse _


theorem pathGraph_adj {n : ℕ} {u v : Fin n} :
    (pathGraph n).Adj u v ↔ u.val + 1 = v.val ∨ v.val + 1 = u.val := by
  /-
    n : Nat
    u v : Fin n
    ⊢ Iff ((SimpleGraph.pathGraph n).Adj u v) (Or (Eq (HAdd.hAdd (↑u) 1) ↑v) (Eq ( …
  -/
  simp only [pathGraph, hasse]
  /-
    n : Nat
    u v : Fin n
    ⊢ Iff (Or (CovBy u v) (CovBy v u)) (Or (Eq (HAdd.hAdd (↑u) 1) ↑v) (Eq (HAdd.hA …
  -/
  simp_rw [← Fin.coe_covBy_iff, covBy_iff_add_one_eq]
  /-
    🎉 no goals
  -/


theorem pathGraph_preconnected (n : ℕ) : (pathGraph n).Preconnected :=
  hasse_preconnected_of_succ _


theorem pathGraph_connected (n : ℕ) : (pathGraph (n + 1)).Connected :=
  ⟨pathGraph_preconnected _⟩


theorem pathGraph_two_eq_top : pathGraph 2 = ⊤ := by
  /-
    ⊢ Eq (SimpleGraph.pathGraph 2) Top.top
  -/
  ext u v
  /-
    case Adj.h.h.a
    u v : Fin 2
    ⊢ Iff ((SimpleGraph.pathGraph 2).Adj u v) (Top.top.Adj u v)
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
  fin_cases u <;> fin_cases v <;> simp [pathGraph, ← Fin.coe_covBy_iff, covBy_iff_add_one_eq]
                                  /-
                                    🎉 no goals
                                  -/


