theorem List.sum_smul {l : List R} {x : M} : l.sum • x = (l.map fun r ↦ r • x).sum :=
  map_list_sum ((smulAddHom R M).flip x) l


theorem Multiset.sum_smul {l : Multiset R} {x : M} : l.sum • x = (l.map fun r ↦ r • x).sum :=
  ((smulAddHom R M).flip x).map_multiset_sum l


theorem Multiset.sum_smul_sum {s : Multiset R} {t : Multiset M} :
    s.sum • t.sum = ((s ×ˢ t).map fun p : R × M ↦ p.fst • p.snd).sum := by
  /-
    R : Type u_5
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Multiset R
    t : Multiset M
    ⊢ Eq (HSMul.hSMul s.sum t.sum) (Multiset.map (fun p => HSMul.hSMul p.1 p.2) (S …
  -/
  induction' s using Multiset.induction with a s ih
    /-
      case empty
      R : Type u_5
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      t : Multiset M
      ⊢ Eq (HSMul.hSMul (Multiset.sum 0) t.sum) (Multiset.map (fun p => HSMul.hSMul  …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u_5
      M : Type u_6
      inst✝² : Semiring R
      inst✝¹ : AddCommMonoid M
      inst✝ : Module R M
      t : Multiset M
      a : R
      s : Multiset R
      ih : Eq (HSMul.hSMul s.sum t.sum) (Multiset.map (fun p => HSMul.hSMul p.1 p.2) …
      ⊢ Eq (HSMul.hSMul (Multiset.cons a s).sum t.sum) (Multiset.map (fun p => HSMul …
    -/
  · simp [add_smul, ih, ← Multiset.smul_sum]
    /-
      🎉 no goals
    -/


theorem Finset.sum_smul {f : ι → R} {s : Finset ι} {x : M} :
    (∑ i ∈ s, f i) • x = ∑ i ∈ s, f i • x := map_sum ((smulAddHom R M).flip x) f s


lemma Finset.sum_smul_sum (s : Finset α) (t : Finset β) {f : α → R} {g : β → M} :
    (∑ i ∈ s, f i) • ∑ j ∈ t, g j = ∑ i ∈ s, ∑ j ∈ t, f i • g j := by
  /-
    α : Type u_3
    β : Type u_4
    R : Type u_5
    M : Type u_6
    inst✝² : Semiring R
    inst✝¹ : AddCommMonoid M
    inst✝ : Module R M
    s : Finset α
    t : Finset β
    f : α → R
    g : β → M
    ⊢ Eq (HSMul.hSMul (s.sum fun i => f i) (t.sum fun j => g j)) (s.sum fun i => t …
  -/
  simp_rw [sum_smul, ← smul_sum]
  /-
    🎉 no goals
  -/


lemma Fintype.sum_smul_sum [Fintype α] [Fintype β] (f : α → R) (g : β → M) :
    (∑ i, f i) • ∑ j, g j = ∑ i, ∑ j, f i • g j := Finset.sum_smul_sum _ _


theorem Finset.cast_card [CommSemiring R] (s : Finset α) : (#s : R) = ∑ _ ∈ s, 1 := by
  /-
    α : Type u_3
    R : Type u_5
    inst✝ : CommSemiring R
    s : Finset α
    ⊢ Eq (↑s.card) (s.sum fun x => 1)
  -/
  rw [Finset.sum_const, Nat.smul_one_eq_cast]
  /-
    🎉 no goals
  -/


lemma sum_piFinset_apply (f : κ → α) (s : Finset κ) (i : ι) :
    ∑ g ∈ piFinset fun _ : ι ↦ s, f (g i) = #s ^ (card ι - 1) • ∑ b ∈ s, f b := by
  classical
  rw [Finset.sum_comp]
  simp only [eval_image_piFinset_const, card_filter_piFinset_const s, ite_smul, zero_smul, smul_sum,
    Finset.sum_ite_mem, inter_self]


