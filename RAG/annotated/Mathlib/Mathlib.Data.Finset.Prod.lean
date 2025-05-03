/-- `product s t` is the set of pairs `(a, b)` such that `a ∈ s` and `b ∈ t`. -/
protected def product (s : Finset α) (t : Finset β) : Finset (α × β) :=
  ⟨_, s.nodup.product t.nodup⟩


instance instSProd : SProd (Finset α) (Finset β) (Finset (α × β)) where
  sprod := Finset.product


@[simp]
theorem product_val : (s ×ˢ t).1 = s.1 ×ˢ t.1 :=
  rfl


@[simp]
theorem mem_product {p : α × β} : p ∈ s ×ˢ t ↔ p.1 ∈ s ∧ p.2 ∈ t :=
  Multiset.mem_product


theorem mk_mem_product (ha : a ∈ s) (hb : b ∈ t) : (a, b) ∈ s ×ˢ t :=
  mem_product.2 ⟨ha, hb⟩


@[simp, norm_cast]
theorem coe_product (s : Finset α) (t : Finset β) :
    (↑(s ×ˢ t) : Set (α × β)) = (s : Set α) ×ˢ t :=
  Set.ext fun _ => Finset.mem_product


theorem subset_product_image_fst [DecidableEq α] : (s ×ˢ t).image Prod.fst ⊆ s := fun i => by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq α
    i : α
    ⊢ Membership.mem (Finset.image Prod.fst (SProd.sprod s t)) i → Membership.mem  …
  -/
  simp +contextual [mem_image]
  /-
    🎉 no goals
  -/


theorem subset_product_image_snd [DecidableEq β] : (s ×ˢ t).image Prod.snd ⊆ t := fun i => by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq β
    i : β
    ⊢ Membership.mem (Finset.image Prod.snd (SProd.sprod s t)) i → Membership.mem  …
  -/
  simp +contextual [mem_image]
  /-
    🎉 no goals
  -/


theorem product_image_fst [DecidableEq α] (ht : t.Nonempty) : (s ×ˢ t).image Prod.fst = s := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq α
    ht : t.Nonempty
    ⊢ Eq (Finset.image Prod.fst (SProd.sprod s t)) s
  -/
  ext i
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq α
    ht : t.Nonempty
    i : α
    ⊢ Iff (Membership.mem (Finset.image Prod.fst (SProd.sprod s t)) i) (Membership …
  -/
  simp [mem_image, ht.exists_mem]
  /-
    🎉 no goals
  -/


theorem product_image_snd [DecidableEq β] (ht : s.Nonempty) : (s ×ˢ t).image Prod.snd = t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq β
    ht : s.Nonempty
    ⊢ Eq (Finset.image Prod.snd (SProd.sprod s t)) t
  -/
  ext i
  /-
    case h
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    inst✝ : DecidableEq β
    ht : s.Nonempty
    i : β
    ⊢ Iff (Membership.mem (Finset.image Prod.snd (SProd.sprod s t)) i) (Membership …
  -/
  simp [mem_image, ht.exists_mem]
  /-
    🎉 no goals
  -/


theorem subset_product [DecidableEq α] [DecidableEq β] {s : Finset (α × β)} :
    s ⊆ s.image Prod.fst ×ˢ s.image Prod.snd := fun _ hp =>
  mem_product.2 ⟨mem_image_of_mem _ hp, mem_image_of_mem _ hp⟩


@[gcongr]
theorem product_subset_product (hs : s ⊆ s') (ht : t ⊆ t') : s ×ˢ t ⊆ s' ×ˢ t' := fun ⟨_, _⟩ h =>
  mem_product.2 ⟨hs (mem_product.1 h).1, ht (mem_product.1 h).2⟩


@[gcongr]
theorem product_subset_product_left (hs : s ⊆ s') : s ×ˢ t ⊆ s' ×ˢ t :=
  product_subset_product hs (Subset.refl _)


@[gcongr]
theorem product_subset_product_right (ht : t ⊆ t') : s ×ˢ t ⊆ s ×ˢ t' :=
  product_subset_product (Subset.refl _) ht


theorem map_swap_product (s : Finset α) (t : Finset β) :
    (t ×ˢ s).map ⟨Prod.swap, Prod.swap_injective⟩ = s ×ˢ t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_2
      s : Finset α
      t : Finset β
      ⊢ Eq ↑(Finset.map { toFun := Prod.swap, inj' := ⋯ } (SProd.sprod t s)) ↑(SProd …
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_2
      s : Finset α
      t : Finset β
      ⊢ Eq (Set.image (⇑{ toFun := Prod.swap, inj' := ⋯ }) (SProd.sprod ↑t ↑s)) (SPr …
    -/
    exact Set.image_swap_prod _ _
    /-
      🎉 no goals
    -/


@[simp]
theorem image_swap_product [DecidableEq (α × β)] (s : Finset α) (t : Finset β) :
    (t ×ˢ s).image Prod.swap = s ×ˢ t :=
  coe_injective <| by
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq (Prod α β)
      s : Finset α
      t : Finset β
      ⊢ Eq ↑(Finset.image Prod.swap (SProd.sprod t s)) ↑(SProd.sprod s t)
    -/
    push_cast
    /-
      α : Type u_1
      β : Type u_2
      inst✝ : DecidableEq (Prod α β)
      s : Finset α
      t : Finset β
      ⊢ Eq (Set.image Prod.swap (SProd.sprod ↑t ↑s)) (SProd.sprod ↑s ↑t)
    -/
    exact Set.image_swap_prod _ _
    /-
      🎉 no goals
    -/


theorem product_eq_biUnion [DecidableEq (α × β)] (s : Finset α) (t : Finset β) :
    s ×ˢ t = s.biUnion fun a => t.image fun b => (a, b) :=
  ext fun ⟨x, y⟩ => by
    simp only [mem_product, mem_biUnion, mem_image, exists_prop, Prod.mk.inj_iff, and_left_comm,
      exists_and_left, exists_eq_right, exists_eq_left]


theorem product_eq_biUnion_right [DecidableEq (α × β)] (s : Finset α) (t : Finset β) :
    s ×ˢ t = t.biUnion fun b => s.image fun a => (a, b) :=
  ext fun ⟨x, y⟩ => by
    simp only [mem_product, mem_biUnion, mem_image, exists_prop, Prod.mk.inj_iff, and_left_comm,
      exists_and_left, exists_eq_right, exists_eq_left]


/-- See also `Finset.sup_product_left`. -/
@[simp]
theorem product_biUnion [DecidableEq γ] (s : Finset α) (t : Finset β) (f : α × β → Finset γ) :
    (s ×ˢ t).biUnion f = s.biUnion fun a => t.biUnion fun b => f (a, b) := by
  /-
    α : Type u_1
    β : Type u_2
    γ : Type u_3
    inst✝ : DecidableEq γ
    s : Finset α
    t : Finset β
    f : Prod α β → Finset γ
    ⊢ Eq ((SProd.sprod s t).biUnion f) (s.biUnion fun a => t.biUnion fun b => f {  …
  -/
  classical simp_rw [product_eq_biUnion, biUnion_biUnion, image_biUnion]
  /-
    🎉 no goals
  -/


@[simp]
theorem card_product (s : Finset α) (t : Finset β) : card (s ×ˢ t) = card s * card t :=
  Multiset.card_product _ _


/-- The product of two Finsets is nontrivial iff both are nonempty
  at least one of them is nontrivial. -/
lemma nontrivial_prod_iff : (s ×ˢ t).Nontrivial ↔
    s.Nonempty ∧ t.Nonempty ∧ (s.Nontrivial ∨ t.Nontrivial) := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    ⊢ Iff (SProd.sprod s t).Nontrivial (And s.Nonempty (And t.Nonempty (Or s.Nontr …
  -/
  simp_rw [← card_pos, ← one_lt_card_iff_nontrivial, card_product]; apply Nat.one_lt_mul_iff
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem filter_product (p : α → Prop) (q : β → Prop) [DecidablePred p] [DecidablePred q] :
    ((s ×ˢ t).filter fun x : α × β => p x.1 ∧ q x.2) = s.filter p ×ˢ t.filter q := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    p : α → Prop
    q : β → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    ⊢ Eq (Finset.filter (fun x => And (p x.1) (q x.2)) (SProd.sprod s t)) (SProd.s …
  -/
  ext ⟨a, b⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    p : α → Prop
    q : β → Prop
    inst✝¹ : DecidablePred p
    inst✝ : DecidablePred q
    a : α
    b : β
    ⊢ Iff (Membership.mem (Finset.filter (fun x => And (p x.1) (q x.2)) (SProd.spr …
  -/
  simp [mem_filter, mem_product, decide_eq_true_eq, and_comm, and_left_comm, and_assoc]
  /-
    🎉 no goals
  -/


theorem filter_product_left (p : α → Prop) [DecidablePred p] :
    ((s ×ˢ t).filter fun x : α × β => p x.1) = s.filter p ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    p : α → Prop
    inst✝ : DecidablePred p
    ⊢ Eq (Finset.filter (fun x => p x.1) (SProd.sprod s t)) (SProd.sprod (Finset.f …
  -/
  simpa using filter_product p fun _ => true
  /-
    🎉 no goals
  -/


theorem filter_product_right (q : β → Prop) [DecidablePred q] :
    ((s ×ˢ t).filter fun x : α × β => q x.2) = s ×ˢ t.filter q := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t : Finset β
    q : β → Prop
    inst✝ : DecidablePred q
    ⊢ Eq (Finset.filter (fun x => q x.2) (SProd.sprod s t)) (SProd.sprod s (Finset …
  -/
  simpa using filter_product (fun _ : α => true) q
  /-
    🎉 no goals
  -/


theorem filter_product_card (s : Finset α) (t : Finset β) (p : α → Prop) (q : β → Prop)
    [DecidablePred p] [DecidablePred q] :
    ((s ×ˢ t).filter fun x : α × β => (p x.1) = (q x.2)).card =
      (s.filter p).card * (t.filter q).card +
        (s.filter (¬ p ·)).card * (t.filter (¬ q ·)).card := by
  classical
  rw [← card_product, ← card_product, ← filter_product, ← filter_product, ← card_union_of_disjoint]
  · apply congr_arg
    ext ⟨a, b⟩
    simp only [filter_union_right, mem_filter, mem_product]
    constructor <;> intro h <;> use h.1
    · simp only [h.2, Function.comp_apply, Decidable.em, and_self]
    · revert h
      simp only [Function.comp_apply, and_imp]
      rintro _ _ (_|_) <;> simp [*]
  · apply Finset.disjoint_filter_filter'
    exact (disjoint_compl_right.inf_left _).inf_right _


@[simp]
theorem empty_product (t : Finset β) : (∅ : Finset α) ×ˢ t = ∅ :=
  rfl


@[simp]
theorem product_empty (s : Finset α) : s ×ˢ (∅ : Finset β) = ∅ :=
  eq_empty_of_forall_not_mem fun _ h => not_mem_empty _ (Finset.mem_product.1 h).2


@[aesop safe apply (rule_sets := [finsetNonempty])]
theorem Nonempty.product (hs : s.Nonempty) (ht : t.Nonempty) : (s ×ˢ t).Nonempty :=
  let ⟨x, hx⟩ := hs
  let ⟨y, hy⟩ := ht
  ⟨(x, y), mem_product.2 ⟨hx, hy⟩⟩


theorem Nonempty.fst (h : (s ×ˢ t).Nonempty) : s.Nonempty :=
  let ⟨xy, hxy⟩ := h
  ⟨xy.1, (mem_product.1 hxy).1⟩


theorem Nonempty.snd (h : (s ×ˢ t).Nonempty) : t.Nonempty :=
  let ⟨xy, hxy⟩ := h
  ⟨xy.2, (mem_product.1 hxy).2⟩


@[simp]
theorem nonempty_product : (s ×ˢ t).Nonempty ↔ s.Nonempty ∧ t.Nonempty :=
  ⟨fun h => ⟨h.fst, h.snd⟩, fun h => h.1.product h.2⟩


@[simp]
theorem product_eq_empty {s : Finset α} {t : Finset β} : s ×ˢ t = ∅ ↔ s = ∅ ∨ t = ∅ := by
  rw [← not_nonempty_iff_eq_empty, nonempty_product, not_and_or, not_nonempty_iff_eq_empty,
    not_nonempty_iff_eq_empty]


@[simp]
theorem singleton_product {a : α} :
    ({a} : Finset α) ×ˢ t = t.map ⟨Prod.mk a, Prod.mk.inj_left _⟩ := by
  /-
    α : Type u_1
    β : Type u_2
    t : Finset β
    a : α
    ⊢ Eq (SProd.sprod (Singleton.singleton a) t) (Finset.map { toFun := Prod.mk a, …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    t : Finset β
    a x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod (Singleton.singleton a) t) { fst := x, snd  …
  -/
  simp [and_left_comm, eq_comm]
  /-
    🎉 no goals
  -/


@[simp]
theorem product_singleton {b : β} : s ×ˢ {b} = s.map ⟨fun i => (i, b), Prod.mk.inj_right _⟩ := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    b : β
    ⊢ Eq (SProd.sprod s (Singleton.singleton b)) (Finset.map { toFun := fun i => { …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Finset α
    b : β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod s (Singleton.singleton b)) { fst := x, snd  …
  -/
  simp [and_left_comm, eq_comm]
  /-
    🎉 no goals
  -/


theorem singleton_product_singleton {a : α} {b : β} :
    ({a} ×ˢ {b} : Finset _) = {(a, b)} := by
  /-
    α : Type u_1
    β : Type u_2
    a : α
    b : β
    ⊢ Eq (SProd.sprod (Singleton.singleton a) (Singleton.singleton b)) (Singleton. …
  -/
  simp only [product_singleton, Function.Embedding.coeFn_mk, map_singleton]
  /-
    🎉 no goals
  -/


@[simp]
theorem union_product [DecidableEq α] [DecidableEq β] : (s ∪ s') ×ˢ t = s ×ˢ t ∪ s' ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s s' : Finset α
    t : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (SProd.sprod (Union.union s s') t) (Union.union (SProd.sprod s t) (SProd. …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s s' : Finset α
    t : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod (Union.union s s') t) { fst := x, snd := y  …
  -/
  simp only [or_and_right, mem_union, mem_product]
  /-
    🎉 no goals
  -/


@[simp]
theorem product_union [DecidableEq α] [DecidableEq β] : s ×ˢ (t ∪ t') = s ×ˢ t ∪ s ×ˢ t' := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t t' : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (SProd.sprod s (Union.union t t')) (Union.union (SProd.sprod s t) (SProd. …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Finset α
    t t' : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod s (Union.union t t')) { fst := x, snd := y  …
  -/
  simp only [and_or_left, mem_union, mem_product]
  /-
    🎉 no goals
  -/


theorem inter_product [DecidableEq α] [DecidableEq β] : (s ∩ s') ×ˢ t = s ×ˢ t ∩ s' ×ˢ t := by
  /-
    α : Type u_1
    β : Type u_2
    s s' : Finset α
    t : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (SProd.sprod (Inter.inter s s') t) (Inter.inter (SProd.sprod s t) (SProd. …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s s' : Finset α
    t : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod (Inter.inter s s') t) { fst := x, snd := y  …
  -/
  simp only [← and_and_right, mem_inter, mem_product]
  /-
    🎉 no goals
  -/


theorem product_inter [DecidableEq α] [DecidableEq β] : s ×ˢ (t ∩ t') = s ×ˢ t ∩ s ×ˢ t' := by
  /-
    α : Type u_1
    β : Type u_2
    s : Finset α
    t t' : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (SProd.sprod s (Inter.inter t t')) (Inter.inter (SProd.sprod s t) (SProd. …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s : Finset α
    t t' : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    x : α
    y : β
    ⊢ Iff (Membership.mem (SProd.sprod s (Inter.inter t t')) { fst := x, snd := y  …
  -/
  simp only [← and_and_left, mem_inter, mem_product]
  /-
    🎉 no goals
  -/


theorem product_inter_product [DecidableEq α] [DecidableEq β] :
    s ×ˢ t ∩ s' ×ˢ t' = (s ∩ s') ×ˢ (t ∩ t') := by
  /-
    α : Type u_1
    β : Type u_2
    s s' : Finset α
    t t' : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    ⊢ Eq (Inter.inter (SProd.sprod s t) (SProd.sprod s' t')) (SProd.sprod (Inter.i …
  -/
  ext ⟨x, y⟩
  /-
    case h.mk
    α : Type u_1
    β : Type u_2
    s s' : Finset α
    t t' : Finset β
    inst✝¹ : DecidableEq α
    inst✝ : DecidableEq β
    x : α
    y : β
    ⊢ Iff (Membership.mem (Inter.inter (SProd.sprod s t) (SProd.sprod s' t')) { fs …
  -/
  simp only [and_assoc, and_left_comm, mem_inter, mem_product]
  /-
    🎉 no goals
  -/


theorem disjoint_product : Disjoint (s ×ˢ t) (s' ×ˢ t') ↔ Disjoint s s' ∨ Disjoint t t' := by
  /-
    α : Type u_1
    β : Type u_2
    s s' : Finset α
    t t' : Finset β
    ⊢ Iff (Disjoint (SProd.sprod s t) (SProd.sprod s' t')) (Or (Disjoint s s') (Di …
  -/
  simp_rw [← disjoint_coe, coe_product, Set.disjoint_prod]
  /-
    🎉 no goals
  -/


@[simp]
theorem disjUnion_product (hs : Disjoint s s') :
    s.disjUnion s' hs ×ˢ t = (s ×ˢ t).disjUnion (s' ×ˢ t) (disjoint_product.mpr <| Or.inl hs) :=
  eq_of_veq <| Multiset.add_product _ _ _


@[simp]
theorem product_disjUnion (ht : Disjoint t t') :
    s ×ˢ t.disjUnion t' ht = (s ×ˢ t).disjUnion (s ×ˢ t') (disjoint_product.mpr <| Or.inr ht) :=
  eq_of_veq <| Multiset.product_add _ _ _


/-- Given a finite set `s`, the diagonal, `s.diag` is the set of pairs of the form `(a, a)` for
`a ∈ s`. -/
def diag :=
  (s ×ˢ s).filter fun a : α × α => a.fst = a.snd


/-- Given a finite set `s`, the off-diagonal, `s.offDiag` is the set of pairs `(a, b)` with `a ≠ b`
for `a, b ∈ s`. -/
def offDiag :=
  (s ×ˢ s).filter fun a : α × α => a.fst ≠ a.snd


@[simp]
theorem mem_diag : x ∈ s.diag ↔ x.1 ∈ s ∧ x.1 = x.2 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x : Prod α α
    ⊢ Iff (Membership.mem s.diag x) (And (Membership.mem s x.1) (Eq x.1 x.2))
  -/
  simp +contextual [diag]
  /-
    🎉 no goals
  -/


@[simp]
theorem mem_offDiag : x ∈ s.offDiag ↔ x.1 ∈ s ∧ x.2 ∈ s ∧ x.1 ≠ x.2 := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    x : Prod α α
    ⊢ Iff (Membership.mem s.offDiag x) (And (Membership.mem s x.1) (And (Membershi …
  -/
  simp [offDiag, and_assoc]
  /-
    🎉 no goals
  -/


@[simp, norm_cast]
theorem coe_offDiag : (s.offDiag : Set (α × α)) = (s : Set α).offDiag :=
  Set.ext fun _ => mem_offDiag


@[simp]
theorem diag_card : (diag s).card = s.card := by
  suffices diag s = s.image fun a => (a, a) by
    rw [this]
    apply card_image_of_injOn
    exact fun x1 _ x2 _ h3 => (Prod.mk.inj h3).1
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq s.diag (Finset.image (fun a => { fst := a, snd := a }) s)
  -/
  ext ⟨a₁, a₂⟩
  /-
    case h.mk
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a₁ a₂ : α
    ⊢ Iff (Membership.mem s.diag { fst := a₁, snd := a₂ }) (Membership.mem (Finset …
  -/
  rw [mem_diag]
  /-
    case h.mk
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a₁ a₂ : α
    ⊢ Iff (And (Membership.mem s { fst := a₁, snd := a₂ }.1) (Eq { fst := a₁, snd  …
  -/
  constructor <;> intro h <;> rw [Finset.mem_image] at *
    /-
      case h.mk.mp
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a₁ a₂ : α
      h : And (Membership.mem s { fst := a₁, snd := a₂ }.1) (Eq { fst := a₁, snd :=  …
      ⊢ Exists fun a => And (Membership.mem s a) (Eq { fst := a, snd := a } { fst := …
    -/
  · use a₁
    /-
      case h
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a₁ a₂ : α
      h : And (Membership.mem s { fst := a₁, snd := a₂ }.1) (Eq { fst := a₁, snd :=  …
      ⊢ And (Membership.mem s a₁) (Eq { fst := a₁, snd := a₁ } { fst := a₁, snd := a …
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case h.mk.mpr
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a₁ a₂ : α
      h : Exists fun a => And (Membership.mem s a) (Eq { fst := a, snd := a } { fst  …
      ⊢ And (Membership.mem s { fst := a₁, snd := a₂ }.1) (Eq { fst := a₁, snd := a₂ …
    -/
  · rcases h with ⟨a, h1, h2⟩
    /-
      case h.mk.mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a₁ a₂ a : α
      h1 : Membership.mem s a
      h2 : Eq { fst := a, snd := a } { fst := a₁, snd := a₂ }
      ⊢ And (Membership.mem s { fst := a₁, snd := a₂ }.1) (Eq { fst := a₁, snd := a₂ …
    -/
    have h := Prod.mk.inj h2
    /-
      case h.mk.mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a₁ a₂ a : α
      h1 : Membership.mem s a
      h2 : Eq { fst := a, snd := a } { fst := a₁, snd := a₂ }
      h : And (Eq a a₁) (Eq a a₂)
      ⊢ And (Membership.mem s { fst := a₁, snd := a₂ }.1) (Eq { fst := a₁, snd := a₂ …
    -/
    rw [← h.1, ← h.2]
    /-
      case h.mk.mpr.intro.intro
      α : Type u_1
      inst✝ : DecidableEq α
      s : Finset α
      a₁ a₂ a : α
      h1 : Membership.mem s a
      h2 : Eq { fst := a, snd := a } { fst := a₁, snd := a₂ }
      h : And (Eq a a₁) (Eq a a₂)
      ⊢ And (Membership.mem s { fst := a, snd := a }.1) (Eq { fst := a, snd := a }.1 …
    -/
    use h1
    /-
      🎉 no goals
    -/


@[simp]
theorem offDiag_card : (offDiag s).card = s.card * s.card - s.card :=
                                                                 /-
                                                                   α : Type u_1
                                                                   inst✝ : DecidableEq α
                                                                   s : Finset α
                                                                   this : Eq (HAdd.hAdd s.diag.card s.offDiag.card) (HMul.hMul s.card s.card)
                                                                   ⊢ Eq s.offDiag.card (HSub.hSub (HMul.hMul s.card s.card) s.card)
                                                                 -/
     /-
       α : Type u_1
       inst✝ : DecidableEq α
       s : Finset α
       ⊢ Eq (HAdd.hAdd s.diag.card s.offDiag.card) (HMul.hMul s.card s.card)
     -/
  suffices (diag s).card + (offDiag s).card = s.card * s.card by rw [s.diag_card] at this; omega
     /-
       α : Type u_1
       inst✝ : DecidableEq α
       s : Finset α
       ⊢ Eq (HAdd.hAdd (Finset.filter (fun a => Eq a.1 a.2) (SProd.sprod s s)).card ( …
     -/
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/
     /-
       🎉 no goals
     -/
  by rw [← card_product, diag, offDiag]
     conv_rhs => rw [← filter_card_add_filter_neg_card_eq_card (fun a => a.1 = a.2)]


@[mono]
theorem diag_mono : Monotone (diag : Finset α → Finset (α × α)) := fun _ _ h _ hx =>
  mem_diag.2 <| And.imp_left (@h _) <| mem_diag.1 hx


@[mono]
theorem offDiag_mono : Monotone (offDiag : Finset α → Finset (α × α)) := fun _ _ h _ hx =>
  mem_offDiag.2 <| And.imp (@h _) (And.imp_left <| @h _) <| mem_offDiag.1 hx


@[simp]
theorem diag_empty : (∅ : Finset α).diag = ∅ :=
  rfl


@[simp]
theorem offDiag_empty : (∅ : Finset α).offDiag = ∅ :=
  rfl


@[simp]
theorem diag_union_offDiag : s.diag ∪ s.offDiag = s ×ˢ s := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Union.union s.diag s.offDiag) (SProd.sprod s s)
  -/
  conv_rhs => rw [← filter_union_filter_neg_eq (fun a => a.1 = a.2) (s ×ˢ s)]
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (Union.union s.diag s.offDiag) (Union.union (Finset.filter (fun a => Eq a …
  -/
  rfl
  /-
    🎉 no goals
  -/


@[simp]
theorem disjoint_diag_offDiag : Disjoint s.diag s.offDiag :=
  disjoint_filter_filter_neg (s ×ˢ s) (s ×ˢ s) (fun a => a.1 = a.2)


theorem product_sdiff_diag : s ×ˢ s \ s.diag = s.offDiag := by
  rw [← diag_union_offDiag, union_comm, union_sdiff_self,
    sdiff_eq_self_of_disjoint (disjoint_diag_offDiag _).symm]


theorem product_sdiff_offDiag : s ×ˢ s \ s.offDiag = s.diag := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    ⊢ Eq (SDiff.sdiff (SProd.sprod s s) s.offDiag) s.diag
  -/
  rw [← diag_union_offDiag, union_sdiff_self, sdiff_eq_self_of_disjoint (disjoint_diag_offDiag _)]
  /-
    🎉 no goals
  -/


theorem diag_inter : (s ∩ t).diag = s.diag ∩ t.diag :=
                  /-
                    α : Type u_1
                    inst✝ : DecidableEq α
                    s t : Finset α
                    x : Prod α α
                    ⊢ Iff (Membership.mem (Inter.inter s t).diag x) (Membership.mem (Inter.inter s …
                  -/
  ext fun x => by simpa only [mem_diag, mem_inter] using and_and_right
                  /-
                    🎉 no goals
                  -/


theorem offDiag_inter : (s ∩ t).offDiag = s.offDiag ∩ t.offDiag :=
  coe_injective <| by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      ⊢ Eq ↑(Inter.inter s t).offDiag ↑(Inter.inter s.offDiag t.offDiag)
    -/
    push_cast
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      ⊢ Eq (Inter.inter ↑s ↑t).offDiag (Inter.inter (↑s).offDiag (↑t).offDiag)
    -/
    exact Set.offDiag_inter _ _
    /-
      🎉 no goals
    -/


theorem diag_union : (s ∪ t).diag = s.diag ∪ t.diag := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    ⊢ Eq (Union.union s t).diag (Union.union s.diag t.diag)
  -/
  ext ⟨i, j⟩
  /-
    case h.mk
    α : Type u_1
    inst✝ : DecidableEq α
    s t : Finset α
    i j : α
    ⊢ Iff (Membership.mem (Union.union s t).diag { fst := i, snd := j }) (Membersh …
  -/
  simp only [mem_diag, mem_union, or_and_right]
  /-
    🎉 no goals
  -/


theorem offDiag_union (h : Disjoint s t) :
    (s ∪ t).offDiag = s.offDiag ∪ t.offDiag ∪ s ×ˢ t ∪ t ×ˢ s :=
  coe_injective <| by
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : Disjoint s t
      ⊢ Eq ↑(Union.union s t).offDiag ↑(Union.union (Union.union (Union.union s.offD …
    -/
    push_cast
    /-
      α : Type u_1
      inst✝ : DecidableEq α
      s t : Finset α
      h : Disjoint s t
      ⊢ Eq (Union.union ↑s ↑t).offDiag (Union.union (Union.union (Union.union (↑s).o …
    -/
    exact Set.offDiag_union (disjoint_coe.2 h)
    /-
      🎉 no goals
    -/


@[simp]
                                                               /-
                                                                 α : Type u_1
                                                                 inst✝ : DecidableEq α
                                                                 a : α
                                                                 ⊢ Eq (Singleton.singleton a).offDiag EmptyCollection.emptyCollection
                                                               -/
theorem offDiag_singleton : ({a} : Finset α).offDiag = ∅ := by simp [← Finset.card_eq_zero]
                                                               /-
                                                                 🎉 no goals
                                                               -/


theorem diag_singleton : ({a} : Finset α).diag = {(a, a)} := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    a : α
    ⊢ Eq (Singleton.singleton a).diag (Singleton.singleton { fst := a, snd := a })
  -/
  rw [← product_sdiff_offDiag, offDiag_singleton, sdiff_empty, singleton_product_singleton]
  /-
    🎉 no goals
  -/


theorem diag_insert : (insert a s).diag = insert (a, a) s.diag := by
  /-
    α : Type u_1
    inst✝ : DecidableEq α
    s : Finset α
    a : α
    ⊢ Eq (Insert.insert a s).diag (Insert.insert { fst := a, snd := a } s.diag)
  -/
  rw [insert_eq, insert_eq, diag_union, diag_singleton]
  /-
    🎉 no goals
  -/


theorem offDiag_insert (has : a ∉ s) : (insert a s).offDiag = s.offDiag ∪ {a} ×ˢ s ∪ s ×ˢ {a} := by
  rw [insert_eq, union_comm, offDiag_union (disjoint_singleton_right.2 has), offDiag_singleton,
    union_empty, union_right_comm]


theorem offDiag_filter_lt_eq_filter_le {ι}
    [PartialOrder ι] [DecidableEq ι]
    [DecidableRel (LE.le (α := ι))] [DecidableRel (LT.lt (α := ι))]
    (s : Finset ι) :
    s.offDiag.filter (fun i => i.1 < i.2) = s.offDiag.filter (fun i => i.1 ≤ i.2) := by
  /-
    ι : Type u_4
    inst✝³ : PartialOrder ι
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableRel LE.le
    inst✝ : DecidableRel LT.lt
    s : Finset ι
    ⊢ Eq (Finset.filter (fun i => LT.lt i.1 i.2) s.offDiag) (Finset.filter (fun i  …
  -/
  rw [Finset.filter_inj']
  /-
    ι : Type u_4
    inst✝³ : PartialOrder ι
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableRel LE.le
    inst✝ : DecidableRel LT.lt
    s : Finset ι
    ⊢ ∀ ⦃a : Prod ι ι⦄, Membership.mem s.offDiag a → Iff (LT.lt a.1 a.2) (LE.le a. …
  -/
  rintro ⟨i, j⟩
  /-
    case mk
    ι : Type u_4
    inst✝³ : PartialOrder ι
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableRel LE.le
    inst✝ : DecidableRel LT.lt
    s : Finset ι
    i j : ι
    ⊢ Membership.mem s.offDiag { fst := i, snd := j } → Iff (LT.lt { fst := i, snd …
  -/
  simp_rw [mem_offDiag, and_imp]
  /-
    case mk
    ι : Type u_4
    inst✝³ : PartialOrder ι
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableRel LE.le
    inst✝ : DecidableRel LT.lt
    s : Finset ι
    i j : ι
    ⊢ Membership.mem s i → Membership.mem s j → Ne i j → Iff (LT.lt i j) (LE.le i j)
  -/
  rintro _ _ h
  /-
    case mk
    ι : Type u_4
    inst✝³ : PartialOrder ι
    inst✝² : DecidableEq ι
    inst✝¹ : DecidableRel LE.le
    inst✝ : DecidableRel LT.lt
    s : Finset ι
    i j : ι
    a✝¹ : Membership.mem s i
    a✝ : Membership.mem s j
    h : Ne i j
    ⊢ Iff (LT.lt i j) (LE.le i j)
  -/
  rw [Ne.le_iff_lt h]
  /-
    🎉 no goals
  -/


