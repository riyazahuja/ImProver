lemma mem_piFinset_iff_zero_tail :
    f ∈ Fintype.piFinset s ↔ f 0 ∈ s 0 ∧ tail f ∈ piFinset (tail s) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    f : (i : Fin (HAdd.hAdd n 1)) → α i
    s : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    ⊢ Iff (Membership.mem (Fintype.piFinset s) f) (And (Membership.mem (s 0) (f 0) …
  -/
  simp only [Fintype.mem_piFinset, forall_fin_succ, tail]
  /-
    🎉 no goals
  -/


lemma mem_piFinset_iff_last_init :
    f ∈ piFinset s ↔ f (last n) ∈ s (last n) ∧ init f ∈ piFinset (init s) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    f : (i : Fin (HAdd.hAdd n 1)) → α i
    s : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    ⊢ Iff (Membership.mem (Fintype.piFinset s) f) (And (Membership.mem (s (Fin.las …
  -/
  simp only [Fintype.mem_piFinset, forall_fin_succ', init, and_comm]
  /-
    🎉 no goals
  -/


lemma mem_piFinset_iff_pivot_removeNth (p : Fin (n + 1)) :
    f ∈ piFinset s ↔ f p ∈ s p ∧ removeNth p f ∈ piFinset (removeNth p s) := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    f : (i : Fin (HAdd.hAdd n 1)) → α i
    s : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    p : Fin (HAdd.hAdd n 1)
    ⊢ Iff (Membership.mem (Fintype.piFinset s) f) (And (Membership.mem (s p) (f p) …
  -/
  simp only [Fintype.mem_piFinset, forall_iff_succAbove p, removeNth]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-09-20")] alias mem_piFinset_succ := mem_piFinset_iff_zero_tail

@[deprecated (since := "2024-09-20")] alias mem_piFinset_succ' := mem_piFinset_iff_last_init


lemma cons_mem_piFinset_cons {x_zero : α 0} {x_tail : (i : Fin n) → α i.succ}
    {s_zero : Finset (α 0)} {s_tail : (i : Fin n) → Finset (α i.succ)} :
    cons x_zero x_tail ∈ piFinset (cons s_zero s_tail) ↔
      x_zero ∈ s_zero ∧ x_tail ∈ piFinset s_tail := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    x_zero : α 0
    x_tail : (i : Fin n) → α i.succ
    s_zero : Finset (α 0)
    s_tail : (i : Fin n) → Finset (α i.succ)
    ⊢ Iff (Membership.mem (Fintype.piFinset (Fin.cons s_zero s_tail)) (Fin.cons x_ …
  -/
  simp_rw [mem_piFinset_iff_zero_tail, cons_zero, tail_cons]
  /-
    🎉 no goals
  -/


lemma snoc_mem_piFinset_snoc {x_last : α (last n)} {x_init : (i : Fin n) → α i.castSucc}
    {s_last : Finset (α (last n))} {s_init : (i : Fin n) → Finset (α i.castSucc)} :
    snoc x_init x_last ∈ piFinset (snoc s_init s_last) ↔
      x_last ∈ s_last ∧ x_init ∈ piFinset s_init := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    x_last : α (Fin.last n)
    x_init : (i : Fin n) → α i.castSucc
    s_last : Finset (α (Fin.last n))
    s_init : (i : Fin n) → Finset (α i.castSucc)
    ⊢ Iff (Membership.mem (Fintype.piFinset (Fin.snoc s_init s_last)) (Fin.snoc x_ …
  -/
  simp_rw [mem_piFinset_iff_last_init, init_snoc, snoc_last]
  /-
    🎉 no goals
  -/


lemma insertNth_mem_piFinset_insertNth {x_pivot : α p} {x_remove : ∀ i, α (succAbove p i)}
    {s_pivot : Finset (α p)} {s_remove : ∀ i, Finset (α (succAbove p i))} :
    insertNth p x_pivot x_remove ∈ piFinset (insertNth p s_pivot s_remove) ↔
      x_pivot ∈ s_pivot ∧ x_remove ∈ piFinset s_remove := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    p : Fin (HAdd.hAdd n 1)
    x_pivot : α p
    x_remove : (i : Fin n) → α (p.succAbove i)
    s_pivot : Finset (α p)
    s_remove : (i : Fin n) → Finset (α (p.succAbove i))
    ⊢ Iff (Membership.mem (Fintype.piFinset (p.insertNth s_pivot s_remove)) (p.ins …
  -/
  simp [mem_piFinset_iff_pivot_removeNth p]
  /-
    🎉 no goals
  -/


lemma map_consEquiv_filter_piFinset (P : (∀ i, α (succ i)) → Prop) [DecidablePred P] :
    {r ∈ piFinset S | P (tail r)}.map (consEquiv α).symm.toEmbedding =
      S 0 ×ˢ {r ∈ piFinset (tail S) | P r} := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α i.succ) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.map (Fin.consEquiv α).symm.toEmbedding (Finset.filter (fun r => P …
  -/
  unfold tail; ext; simp [Fin.forall_iff_succ, and_assoc]
                    /-
                      🎉 no goals
                    -/


lemma map_snocEquiv_filter_piFinset (P : (∀ i, α (castSucc i)) → Prop) [DecidablePred P] :
    {r ∈ piFinset S | P (init r)}.map (snocEquiv α).symm.toEmbedding =
      S (last _) ×ˢ {r ∈ piFinset (init S) | P r} := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α i.castSucc) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.map (Fin.snocEquiv α).symm.toEmbedding (Finset.filter (fun r => P …
  -/
  unfold init; ext; simp [Fin.forall_iff_castSucc, and_assoc]
                    /-
                      🎉 no goals
                    -/


lemma map_insertNthEquiv_filter_piFinset (P : (∀ i, α (p.succAbove i)) → Prop) [DecidablePred P] :
    {r ∈ piFinset S | P (p.removeNth r)}.map (p.insertNthEquiv α).symm.toEmbedding =
      S p ×ˢ {r ∈ piFinset (p.removeNth  S) | P r} := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    p : Fin (HAdd.hAdd n 1)
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α (p.succAbove i)) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.map (Fin.insertNthEquiv α p).symm.toEmbedding (Finset.filter (fun …
  -/
  unfold removeNth; ext; simp [Fin.forall_iff_succAbove p, and_assoc]
                         /-
                           🎉 no goals
                         -/


lemma filter_piFinset_eq_map_consEquiv (P : (∀ i, α (succ i)) → Prop) [DecidablePred P] :
    {r ∈ piFinset S | P (tail r)} =
      (S 0 ×ˢ {r ∈ piFinset (tail S) | P r}).map (consEquiv α).toEmbedding := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α i.succ) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.filter (fun r => P (Fin.tail r)) (Fintype.piFinset S)) (Finset.ma …
  -/
  simp [← map_consEquiv_filter_piFinset, map_map]
  /-
    🎉 no goals
  -/


lemma filter_piFinset_eq_map_snocEquiv (P : (∀ i, α (castSucc i)) → Prop) [DecidablePred P] :
    {r ∈ piFinset S | P (init r)} =
      (S (last _) ×ˢ {r ∈ piFinset (init S) | P r}).map (snocEquiv α).toEmbedding := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α i.castSucc) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.filter (fun r => P (Fin.init r)) (Fintype.piFinset S)) (Finset.ma …
  -/
  simp [← map_snocEquiv_filter_piFinset, map_map]
  /-
    🎉 no goals
  -/


lemma filter_piFinset_eq_map_insertNthEquiv (P : (∀ i, α (p.succAbove i)) → Prop)
    [DecidablePred P] :
    {r ∈ piFinset S | P (p.removeNth r)} =
      (S p ×ˢ {r ∈ piFinset (p.removeNth  S) | P r}).map (p.insertNthEquiv α).toEmbedding := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    p : Fin (HAdd.hAdd n 1)
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α (p.succAbove i)) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.filter (fun r => P (p.removeNth r)) (Fintype.piFinset S)) (Finset …
  -/
  simp [← map_insertNthEquiv_filter_piFinset, map_map]
  /-
    🎉 no goals
  -/


lemma card_consEquiv_filter_piFinset (P : (∀ i, α (succ i)) → Prop) [DecidablePred P] :
    {r ∈ piFinset S | P (tail r)}.card = (S 0).card * {r ∈ piFinset (tail S) | P r}.card := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α i.succ) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.filter (fun r => P (Fin.tail r)) (Fintype.piFinset S)).card (HMul …
  -/
  rw [← card_product, ← map_consEquiv_filter_piFinset, card_map]
  /-
    🎉 no goals
  -/


lemma card_snocEquiv_filter_piFinset (P : (∀ i, α (castSucc i)) → Prop) [DecidablePred P] :
    {r ∈ piFinset S | P (init r)}.card =
      (S (last _)).card * {r ∈ piFinset (init S) | P r}.card := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α i.castSucc) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.filter (fun r => P (Fin.init r)) (Fintype.piFinset S)).card (HMul …
  -/
  rw [← card_product, ← map_snocEquiv_filter_piFinset, card_map]
  /-
    🎉 no goals
  -/


lemma card_insertNthEquiv_filter_piFinset (P : (∀ i, α (p.succAbove i)) → Prop) [DecidablePred P] :
    {r ∈ piFinset S | P (p.removeNth r)}.card =
      (S p).card * {r ∈ piFinset (p.removeNth  S) | P r}.card := by
  /-
    n : Nat
    α : Fin (HAdd.hAdd n 1) → Type u_1
    p : Fin (HAdd.hAdd n 1)
    S : (i : Fin (HAdd.hAdd n 1)) → Finset (α i)
    P : ((i : Fin n) → α (p.succAbove i)) → Prop
    inst✝ : DecidablePred P
    ⊢ Eq (Finset.filter (fun r => P (p.removeNth r)) (Fintype.piFinset S)).card (H …
  -/
  rw [← card_product, ← map_insertNthEquiv_filter_piFinset, card_map]
  /-
    🎉 no goals
  -/


