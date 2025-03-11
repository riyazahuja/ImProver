/-- Disjoint sum of `G` and `H`. -/
@[simps!]
protected def sum (G : SimpleGraph α) (H : SimpleGraph β) : SimpleGraph (α ⊕ β) where
  Adj u v := match u, v with
    | Sum.inl u, Sum.inl v => G.Adj u v
    | Sum.inr u, Sum.inr v => H.Adj u v
    | _, _ => false
  symm u v := match u, v with
    | Sum.inl u, Sum.inl v => G.adj_symm
    | Sum.inr u, Sum.inr v => H.adj_symm
    | Sum.inl _, Sum.inr _ | Sum.inr _, Sum.inl _ => id
                   /-
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     G : SimpleGraph α
                     H : SimpleGraph β
                     u : Sum α β
                     ⊢ Not ((fun u v => SimpleGraph.sum.match_1 (fun u v => Prop) u v (fun u v => G …
                   -/
                               /-
                                 🎉 no goals
                               -/
  loopless u := by cases u <;> simp
                               /-
                                 🎉 no goals
                               -/


@[inherit_doc] infixl:60 " ⊕g " => SimpleGraph.sum


/-- The disjoint sum is commutative up to isomorphism. `Iso.sumComm` as a graph isomorphism. -/
@[simps!]
def Iso.sumComm : G ⊕g H ≃g H ⊕g G := ⟨Equiv.sumComm α β, by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    G : SimpleGraph α
    H : SimpleGraph β
    ⊢ ∀ {a b : Sum α β}, Iff ((H.sum G).Adj ((Equiv.sumComm α β) a) ((Equiv.sumCom …
  -/
  intro u v
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    G : SimpleGraph α
    H : SimpleGraph β
    u v : Sum α β
    ⊢ Iff ((H.sum G).Adj ((Equiv.sumComm α β) u) ((Equiv.sumComm α β) v)) ((G.sum  …
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
  cases u <;> cases v <;> simp⟩
                          /-
                            🎉 no goals
                          -/


/-- The disjoint sum is associative up to isomorphism. `Iso.sumAssoc` as a graph isomorphism. -/
@[simps!]
def Iso.sumAssoc {I : SimpleGraph γ} : (G ⊕g H) ⊕g I ≃g G ⊕g (H ⊕g I) := ⟨Equiv.sumAssoc α β γ, by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    G : SimpleGraph α
    H : SimpleGraph β
    I : SimpleGraph γ
    ⊢ ∀ {a b : Sum (Sum α β) γ}, Iff ((G.sum (H.sum I)).Adj ((Equiv.sumAssoc α β γ …
  -/
  intro u v
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    G : SimpleGraph α
    H : SimpleGraph β
    I : SimpleGraph γ
    u v : Sum (Sum α β) γ
    ⊢ Iff ((G.sum (H.sum I)).Adj ((Equiv.sumAssoc α β γ) u) ((Equiv.sumAssoc α β γ …
  -/
  cases u <;> cases v <;> rename_i u v
    /-
      case inl.inl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      G : SimpleGraph α
      H : SimpleGraph β
      I : SimpleGraph γ
      u v : Sum α β
      ⊢ Iff ((G.sum (H.sum I)).Adj ((Equiv.sumAssoc α β γ) (Sum.inl u)) ((Equiv.sumA …
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
  · cases u <;> cases v <;> simp
                            /-
                              🎉 no goals
                            -/
    /-
      case inl.inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      G : SimpleGraph α
      H : SimpleGraph β
      I : SimpleGraph γ
      u : Sum α β
      v : γ
      ⊢ Iff ((G.sum (H.sum I)).Adj ((Equiv.sumAssoc α β γ) (Sum.inl u)) ((Equiv.sumA …
    -/
                /-
                  🎉 no goals
                -/
  · cases u <;> simp
                /-
                  🎉 no goals
                -/
    /-
      case inr.inl
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      G : SimpleGraph α
      H : SimpleGraph β
      I : SimpleGraph γ
      u : γ
      v : Sum α β
      ⊢ Iff ((G.sum (H.sum I)).Adj ((Equiv.sumAssoc α β γ) (Sum.inr u)) ((Equiv.sumA …
    -/
                /-
                  🎉 no goals
                -/
  · cases v <;> simp
                /-
                  🎉 no goals
                -/
    /-
      case inr.inr
      α : Type u_1
      β : Type u_2
      γ : Type u_3
      G : SimpleGraph α
      H : SimpleGraph β
      I : SimpleGraph γ
      u v : γ
      ⊢ Iff ((G.sum (H.sum I)).Adj ((Equiv.sumAssoc α β γ) (Sum.inr u)) ((Equiv.sumA …
    -/
  · simp⟩
    /-
      🎉 no goals
    -/


/-- The embedding of `G` into `G ⊕g H`. -/
@[simps]
def Embedding.sumInl : G ↪g G ⊕g H where
  toFun u := _root_.Sum.inl u
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   G : SimpleGraph α
                   H : SimpleGraph β
                   u v : α
                   ⊢ Eq ((fun u => Sum.inl u) u) ((fun u => Sum.inl u) v) → Eq u v
                 -/
  inj' u v := by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       G : SimpleGraph α
                       H : SimpleGraph β
                       ⊢ ∀ {a b : α}, Iff ((G.sum H).Adj ({ toFun := fun u => Sum.inl u, inj' := ⋯ }  …
                     -/
  map_rel_iff' := by simp
                     /-
                       🎉 no goals
                     -/


/-- The embedding of `H` into `G ⊕g H`. -/
@[simps]
def Embedding.sumInr : H ↪g G ⊕g H where
  toFun u := _root_.Sum.inr u
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   G : SimpleGraph α
                   H : SimpleGraph β
                   u v : β
                   ⊢ Eq ((fun u => Sum.inr u) u) ((fun u => Sum.inr u) v) → Eq u v
                 -/
  inj' u v := by simp
                 /-
                   🎉 no goals
                 -/
                     /-
                       α : Type u_1
                       β : Type u_2
                       γ : Type u_3
                       G : SimpleGraph α
                       H : SimpleGraph β
                       ⊢ ∀ {a b : β}, Iff ((G.sum H).Adj ({ toFun := fun u => Sum.inr u, inj' := ⋯ }  …
                     -/
  map_rel_iff' := by simp
                     /-
                       🎉 no goals
                     -/


