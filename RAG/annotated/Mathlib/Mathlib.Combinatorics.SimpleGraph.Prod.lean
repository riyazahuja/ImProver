/-- Box product of simple graphs. It relates `(a₁, b)` and `(a₂, b)` if `G` relates `a₁` and `a₂`,
and `(a, b₁)` and `(a, b₂)` if `H` relates `b₁` and `b₂`. -/
def boxProd (G : SimpleGraph α) (H : SimpleGraph β) : SimpleGraph (α × β) where
  Adj x y := G.Adj x.1 y.1 ∧ x.2 = y.2 ∨ H.Adj x.2 y.2 ∧ x.1 = y.1
                 /-
                   α : Type u_1
                   β : Type u_2
                   γ : Type u_3
                   G✝ : SimpleGraph α
                   H✝ : SimpleGraph β
                   G : SimpleGraph α
                   H : SimpleGraph β
                   x y : Prod α β
                   ⊢ (fun x y => Or (And (G.Adj x.1 y.1) (Eq x.2 y.2)) (And (H.Adj x.2 y.2) (Eq x …
                 -/
  symm x y := by simp [and_comm, or_comm, eq_comm, adj_comm]
                 /-
                   🎉 no goals
                 -/
                   /-
                     α : Type u_1
                     β : Type u_2
                     γ : Type u_3
                     G✝ : SimpleGraph α
                     H✝ : SimpleGraph β
                     G : SimpleGraph α
                     H : SimpleGraph β
                     x : Prod α β
                     ⊢ Not ((fun x y => Or (And (G.Adj x.1 y.1) (Eq x.2 y.2)) (And (H.Adj x.2 y.2)  …
                   -/
  loopless x := by simp
                   /-
                     🎉 no goals
                   -/


/-- Box product of simple graphs. It relates `(a₁, b)` and `(a₂, b)` if `G` relates `a₁` and `a₂`,
and `(a, b₁)` and `(a, b₂)` if `H` relates `b₁` and `b₂`. -/
infixl:70 " □ " => boxProd


@[simp]
theorem boxProd_adj {x y : α × β} :
    (G □ H).Adj x y ↔ G.Adj x.1 y.1 ∧ x.2 = y.2 ∨ H.Adj x.2 y.2 ∧ x.1 = y.1 :=
  Iff.rfl


theorem boxProd_adj_left {a₁ : α} {b : β} {a₂ : α} :
    (G □ H).Adj (a₁, b) (a₂, b) ↔ G.Adj a₁ a₂ := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    a₁ : α
    b : β
    a₂ : α
    ⊢ Iff ((G.boxProd H).Adj { fst := a₁, snd := b } { fst := a₂, snd := b }) (G.A …
  -/
  simp only [boxProd_adj, and_true, SimpleGraph.irrefl, false_and, or_false]
  /-
    🎉 no goals
  -/


theorem boxProd_adj_right {a : α} {b₁ b₂ : β} : (G □ H).Adj (a, b₁) (a, b₂) ↔ H.Adj b₁ b₂ := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    a : α
    b₁ b₂ : β
    ⊢ Iff ((G.boxProd H).Adj { fst := a, snd := b₁ } { fst := a, snd := b₂ }) (H.A …
  -/
  simp only [boxProd_adj, SimpleGraph.irrefl, false_and, and_true, false_or]
  /-
    🎉 no goals
  -/


theorem boxProd_neighborSet (x : α × β) :
    (G □ H).neighborSet x = G.neighborSet x.1 ×ˢ {x.2} ∪ {x.1} ×ˢ H.neighborSet x.2 := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    x : Prod α β
    ⊢ Eq ((G.boxProd H).neighborSet x) (Union.union (SProd.sprod (G.neighborSet x. …
  -/
  ext ⟨a', b'⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    x : Prod α β
    a' : α
    b' : β
    ⊢ Iff (Membership.mem ((G.boxProd H).neighborSet x) { fst := a', snd := b' })  …
  -/
  simp only [mem_neighborSet, Set.mem_union, boxProd_adj, Set.mem_prod, Set.mem_singleton_iff]
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    x : Prod α β
    a' : α
    b' : β
    ⊢ Iff (Or (And (G.Adj x.1 a') (Eq x.2 b')) (And (H.Adj x.2 b') (Eq x.1 a'))) ( …
  -/
  simp only [eq_comm, and_comm]
  /-
    🎉 no goals
  -/


/-- The box product is commutative up to isomorphism. `Equiv.prodComm` as a graph isomorphism. -/
@[simps!]
def boxProdComm : G □ H ≃g H □ G := ⟨Equiv.prodComm _ _, or_comm⟩


/-- The box product is associative up to isomorphism. `Equiv.prodAssoc` as a graph isomorphism. -/
@[simps!]
def boxProdAssoc (I : SimpleGraph γ) : G □ H □ I ≃g G □ (H □ I) :=
  ⟨Equiv.prodAssoc _ _ _, fun {x y} => by
    simp only [boxProd_adj, Equiv.prodAssoc_apply, or_and_right, or_assoc, Prod.ext_iff,
      and_assoc, @and_comm (x.fst.fst = _)]⟩


/-- The embedding of `G` into `G □ H` given by `b`. -/
@[simps]
def boxProdLeft (b : β) : G ↪g G □ H where
  toFun a := (a, b)
  inj' _ _ := congr_arg Prod.fst
  map_rel_iff' {_ _} := boxProd_adj_left


/-- The embedding of `H` into `G □ H` given by `a`. -/
@[simps]
def boxProdRight (a : α) : H ↪g G □ H where
  toFun := Prod.mk a
  inj' _ _ := congr_arg Prod.snd
  map_rel_iff' {_ _} := boxProd_adj_right


/-- Turn a walk on `G` into a walk on `G □ H`. -/
protected def boxProdLeft {a₁ a₂ : α} (b : β) : G.Walk a₁ a₂ → (G □ H).Walk (a₁, b) (a₂, b) :=
  Walk.map (G.boxProdLeft H b).toHom


/-- Turn a walk on `H` into a walk on `G □ H`. -/
protected def boxProdRight {b₁ b₂ : β} (a : α) : H.Walk b₁ b₂ → (G □ H).Walk (a, b₁) (a, b₂) :=
  Walk.map (G.boxProdRight H a).toHom


/-- Project a walk on `G □ H` to a walk on `G` by discarding the moves in the direction of `H`. -/
def ofBoxProdLeft [DecidableEq β] [DecidableRel G.Adj] {x y : α × β} :
    (G □ H).Walk x y → G.Walk x.1 y.1
  | nil => nil
  | cons h w =>
    Or.by_cases h
      (fun hG => w.ofBoxProdLeft.cons hG.1)
      (fun hH => hH.2 ▸ w.ofBoxProdLeft)


/-- Project a walk on `G □ H` to a walk on `H` by discarding the moves in the direction of `G`. -/
def ofBoxProdRight [DecidableEq α] [DecidableRel H.Adj] {x y : α × β} :
    (G □ H).Walk x y → H.Walk x.2 y.2
  | nil => nil
  | cons h w =>
    (Or.symm h).by_cases
      (fun hH => w.ofBoxProdRight.cons hH.1)
      (fun hG => hG.2 ▸ w.ofBoxProdRight)


@[simp]
theorem ofBoxProdLeft_boxProdLeft [DecidableEq β] [DecidableRel G.Adj] {a₁ a₂ : α} {b : β} :
    ∀ (w : G.Walk a₁ a₂), (w.boxProdLeft H b).ofBoxProdLeft = w
  | nil => rfl
  | cons' x y z h w => by
    /-
      α : Type u_1
      β : Type u_2
      G : SimpleGraph α
      H : SimpleGraph β
      inst✝¹ : DecidableEq β
      inst✝ : DecidableRel G.Adj
      a₁ a₂ : α
      b : β
      x z y : α
      h : G.Adj x y
      w : G.Walk y z
      ⊢ Eq (SimpleGraph.Walk.boxProdLeft H b (SimpleGraph.Walk.cons' x y z h w)).ofB …
    -/
    rw [Walk.boxProdLeft, map_cons, ofBoxProdLeft, Or.by_cases, dif_pos, ← Walk.boxProdLeft]
      /-
        α : Type u_1
        β : Type u_2
        G : SimpleGraph α
        H : SimpleGraph β
        inst✝¹ : DecidableEq β
        inst✝ : DecidableRel G.Adj
        a₁ a₂ : α
        b : β
        x z y : α
        h : G.Adj x y
        w : G.Walk y z
        ⊢ Eq (SimpleGraph.Walk.cons ⋯ (SimpleGraph.Walk.boxProdLeft H b w).ofBoxProdLe …
      -/
    · simp [ofBoxProdLeft_boxProdLeft]
      /-
        🎉 no goals
      -/
      /-
        case hc
        α : Type u_1
        β : Type u_2
        G : SimpleGraph α
        H : SimpleGraph β
        inst✝¹ : DecidableEq β
        inst✝ : DecidableRel G.Adj
        a₁ a₂ : α
        b : β
        x z y : α
        h : G.Adj x y
        w : G.Walk y z
        ⊢ And (G.Adj { fst := x, snd := b }.1 ((G.boxProdLeft H b).toHom y).1) (Eq { f …
      -/
    · exact ⟨h, rfl⟩
      /-
        🎉 no goals
      -/


@[simp]
theorem ofBoxProdLeft_boxProdRight [DecidableEq α] [DecidableRel G.Adj] {a b₁ b₂ : α} :
    ∀ (w : G.Walk b₁ b₂), (w.boxProdRight G a).ofBoxProdRight = w
  | nil => rfl
  | cons' x y z h w => by
    rw [Walk.boxProdRight, map_cons, ofBoxProdRight, Or.by_cases, dif_pos, ←
      Walk.boxProdRight]
      /-
        α : Type u_1
        G : SimpleGraph α
        inst✝¹ : DecidableEq α
        inst✝ : DecidableRel G.Adj
        a b₁ b₂ x z y : α
        h : G.Adj x y
        w : G.Walk y z
        ⊢ Eq (SimpleGraph.Walk.cons ⋯ (SimpleGraph.Walk.boxProdRight G a w).ofBoxProdR …
      -/
    · simp [ofBoxProdLeft_boxProdRight]
      /-
        🎉 no goals
      -/
      /-
        case hc
        α : Type u_1
        G : SimpleGraph α
        inst✝¹ : DecidableEq α
        inst✝ : DecidableRel G.Adj
        a b₁ b₂ x z y : α
        h : G.Adj x y
        w : G.Walk y z
        ⊢ And (G.Adj { fst := a, snd := x }.2 ((G.boxProdRight G a).toHom y).2) (Eq {  …
      -/
    · exact ⟨h, rfl⟩
      /-
        🎉 no goals
      -/


protected theorem Preconnected.boxProd (hG : G.Preconnected) (hH : H.Preconnected) :
    (G □ H).Preconnected := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    hG : G.Preconnected
    hH : H.Preconnected
    ⊢ (G.boxProd H).Preconnected
  -/
  rintro x y
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    hG : G.Preconnected
    hH : H.Preconnected
    x y : Prod α β
    ⊢ (G.boxProd H).Reachable x y
  -/
  obtain ⟨w₁⟩ := hG x.1 y.1
  /-
    case intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    hG : G.Preconnected
    hH : H.Preconnected
    x y : Prod α β
    w₁ : G.Walk x.1 y.1
    ⊢ (G.boxProd H).Reachable x y
  -/
  obtain ⟨w₂⟩ := hH x.2 y.2
  /-
    case intro.intro
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    hG : G.Preconnected
    hH : H.Preconnected
    x y : Prod α β
    w₁ : G.Walk x.1 y.1
    w₂ : H.Walk x.2 y.2
    ⊢ (G.boxProd H).Reachable x y
  -/
  exact ⟨(w₁.boxProdLeft _ _).append (w₂.boxProdRight _ _)⟩
  /-
    🎉 no goals
  -/


protected theorem Preconnected.ofBoxProdLeft [Nonempty β] (h : (G □ H).Preconnected) :
    G.Preconnected := by
  classical
  rintro a₁ a₂
  obtain ⟨w⟩ := h (a₁, Classical.arbitrary _) (a₂, Classical.arbitrary _)
  exact ⟨w.ofBoxProdLeft⟩


protected theorem Preconnected.ofBoxProdRight [Nonempty α] (h : (G □ H).Preconnected) :
    H.Preconnected := by
  classical
  rintro b₁ b₂
  obtain ⟨w⟩ := h (Classical.arbitrary _, b₁) (Classical.arbitrary _, b₂)
  exact ⟨w.ofBoxProdRight⟩


protected theorem Connected.boxProd (hG : G.Connected) (hH : H.Connected) : (G □ H).Connected := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    hG : G.Connected
    hH : H.Connected
    ⊢ (G.boxProd H).Connected
  -/
  haveI := hG.nonempty
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    hG : G.Connected
    hH : H.Connected
    this : Nonempty α
    ⊢ (G.boxProd H).Connected
  -/
  haveI := hH.nonempty
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    hG : G.Connected
    hH : H.Connected
    this✝ : Nonempty α
    this : Nonempty β
    ⊢ (G.boxProd H).Connected
  -/
  exact ⟨hG.preconnected.boxProd hH.preconnected⟩
  /-
    🎉 no goals
  -/


protected theorem Connected.ofBoxProdLeft (h : (G □ H).Connected) : G.Connected := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    h : (G.boxProd H).Connected
    ⊢ G.Connected
  -/
  haveI := (nonempty_prod.1 h.nonempty).1
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    h : (G.boxProd H).Connected
    this : Nonempty α
    ⊢ G.Connected
  -/
  haveI := (nonempty_prod.1 h.nonempty).2
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    h : (G.boxProd H).Connected
    this✝ : Nonempty α
    this : Nonempty β
    ⊢ G.Connected
  -/
  exact ⟨h.preconnected.ofBoxProdLeft⟩
  /-
    🎉 no goals
  -/


protected theorem Connected.ofBoxProdRight (h : (G □ H).Connected) : H.Connected := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    h : (G.boxProd H).Connected
    ⊢ H.Connected
  -/
  haveI := (nonempty_prod.1 h.nonempty).1
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    h : (G.boxProd H).Connected
    this : Nonempty α
    ⊢ H.Connected
  -/
  haveI := (nonempty_prod.1 h.nonempty).2
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    h : (G.boxProd H).Connected
    this✝ : Nonempty α
    this : Nonempty β
    ⊢ H.Connected
  -/
  exact ⟨h.preconnected.ofBoxProdRight⟩
  /-
    🎉 no goals
  -/


@[simp]
theorem boxProd_connected : (G □ H).Connected ↔ G.Connected ∧ H.Connected :=
  ⟨fun h => ⟨h.ofBoxProdLeft, h.ofBoxProdRight⟩, fun h => h.1.boxProd h.2⟩


instance boxProdFintypeNeighborSet (x : α × β)
    [Fintype (G.neighborSet x.1)] [Fintype (H.neighborSet x.2)] :
    Fintype ((G □ H).neighborSet x) :=
  Fintype.ofEquiv
    ((G.neighborFinset x.1 ×ˢ {x.2}).disjUnion ({x.1} ×ˢ H.neighborFinset x.2) <|
        Finset.disjoint_product.mpr <| Or.inl <| neighborFinset_disjoint_singleton _ _)
    ((Equiv.refl _).subtypeEquiv fun y => by
      simp_rw [Finset.mem_disjUnion, Finset.mem_product, Finset.mem_singleton, mem_neighborFinset,
        mem_neighborSet, Equiv.refl_apply, boxProd_adj]
      /-
        α : Type u_1
        β : Type u_2
        γ : Type u_3
        G : SimpleGraph α
        H : SimpleGraph β
        x : Prod α β
        inst✝¹ : Fintype ↑(G.neighborSet x.1)
        inst✝ : Fintype ↑(H.neighborSet x.2)
        y : Prod α β
        ⊢ Iff (Or (And (G.Adj x.1 y.1) (Eq y.2 x.2)) (And (Eq y.1 x.1) (H.Adj x.2 y.2) …
      -/
      simp only [eq_comm, and_comm])
      /-
        🎉 no goals
      -/


theorem boxProd_neighborFinset (x : α × β)
    [Fintype (G.neighborSet x.1)] [Fintype (H.neighborSet x.2)] [Fintype ((G □ H).neighborSet x)] :
    (G □ H).neighborFinset x =
      (G.neighborFinset x.1 ×ˢ {x.2}).disjUnion ({x.1} ×ˢ H.neighborFinset x.2)
        (Finset.disjoint_product.mpr <| Or.inl <| neighborFinset_disjoint_singleton _ _) := by
  -- swap out the fintype instance for the canonical one
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    x : Prod α β
    inst✝² : Fintype ↑(G.neighborSet x.1)
    inst✝¹ : Fintype ↑(H.neighborSet x.2)
    inst✝ : Fintype ↑((G.boxProd H).neighborSet x)
    ⊢ Eq ((G.boxProd H).neighborFinset x) ((SProd.sprod (G.neighborFinset x.1) (Si …
  -/
  letI : Fintype ((G □ H).neighborSet x) := SimpleGraph.boxProdFintypeNeighborSet _
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    x : Prod α β
    inst✝² : Fintype ↑(G.neighborSet x.1)
    inst✝¹ : Fintype ↑(H.neighborSet x.2)
    inst✝ : Fintype ↑((G.boxProd H).neighborSet x)
    this : Fintype ↑((G.boxProd H).neighborSet x) := SimpleGraph.boxProdFintypeNei …
    ⊢ Eq ((G.boxProd H).neighborFinset x) ((SProd.sprod (G.neighborFinset x.1) (Si …
  -/
  convert_to (G □ H).neighborFinset x = _ using 2
  /-
    case convert_2
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    x : Prod α β
    inst✝² : Fintype ↑(G.neighborSet x.1)
    inst✝¹ : Fintype ↑(H.neighborSet x.2)
    inst✝ : Fintype ↑((G.boxProd H).neighborSet x)
    this : Fintype ↑((G.boxProd H).neighborSet x) := SimpleGraph.boxProdFintypeNei …
    ⊢ Eq ((G.boxProd H).neighborFinset x) ((SProd.sprod (G.neighborFinset x.1) (Si …
  -/
  exact Eq.trans (Finset.map_map _ _ _) Finset.attach_map_val
  /-
    🎉 no goals
  -/


theorem boxProd_degree (x : α × β)
    [Fintype (G.neighborSet x.1)] [Fintype (H.neighborSet x.2)] [Fintype ((G □ H).neighborSet x)] :
    (G □ H).degree x = G.degree x.1 + H.degree x.2 := by
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    x : Prod α β
    inst✝² : Fintype ↑(G.neighborSet x.1)
    inst✝¹ : Fintype ↑(H.neighborSet x.2)
    inst✝ : Fintype ↑((G.boxProd H).neighborSet x)
    ⊢ Eq ((G.boxProd H).degree x) (HAdd.hAdd (G.degree x.1) (H.degree x.2))
  -/
  rw [degree, degree, degree, boxProd_neighborFinset, Finset.card_disjUnion]
  /-
    α : Type u_1
    β : Type u_2
    G : SimpleGraph α
    H : SimpleGraph β
    x : Prod α β
    inst✝² : Fintype ↑(G.neighborSet x.1)
    inst✝¹ : Fintype ↑(H.neighborSet x.2)
    inst✝ : Fintype ↑((G.boxProd H).neighborSet x)
    ⊢ Eq (HAdd.hAdd (SProd.sprod (G.neighborFinset x.1) (Singleton.singleton x.2)) …
  -/
  simp_rw [Finset.card_product, Finset.card_singleton, mul_one, one_mul]
  /-
    🎉 no goals
  -/


