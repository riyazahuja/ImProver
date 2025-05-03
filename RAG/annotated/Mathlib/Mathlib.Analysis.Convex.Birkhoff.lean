/--
If M is a positive scalar multiple of a doubly stochastic matrix, then there is a permutation matrix
whose support is contained in the support of M.
-/
private lemma exists_perm_eq_zero_implies_eq_zero [Nonempty n] {s : R} (hs : 0 < s)
    (hM : ∃ M' ∈ doublyStochastic R n, M = s • M') :
    ∃ σ : Equiv.Perm n, ∀ i j, M i j = 0 → σ.permMatrix R i j = 0 := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : LinearOrderedSemifield R
    M : Matrix n n R
    inst✝ : Nonempty n
    s : R
    hs : LT.lt 0 s
    hM : Exists fun M' => And (Membership.mem (doublyStochastic R n) M') (Eq M (HS …
    ⊢ Exists fun σ => ∀ (i j : n), Eq (M i j) 0 → Eq (Equiv.Perm.permMatrix R σ i  …
  -/
  rw [exists_mem_doublyStochastic_eq_smul_iff hs.le] at hM
  /-
    R : Type u_1
    n : Type u_2
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : LinearOrderedSemifield R
    M : Matrix n n R
    inst✝ : Nonempty n
    s : R
    hs : LT.lt 0 s
    hM : And (∀ (i j : n), LE.le 0 (M i j)) (And (∀ (i : n), Eq (Finset.univ.sum f …
    ⊢ Exists fun σ => ∀ (i j : n), Eq (M i j) 0 → Eq (Equiv.Perm.permMatrix R σ i  …
  -/
  let f (i : n) : Finset n := {j | M i j ≠ 0}
  have hf (A : Finset n) : #A ≤ #(A.biUnion f) := by
    have (i) : ∑ j ∈ f i, M i j = s := by simp [f, sum_subset (filter_subset _ _), hM.2.1]
    have h₁ : ∑ i ∈ A, ∑ j ∈ f i, M i j = #A * s := by simp [this]
    have h₂ : ∑ i, ∑ j ∈ A.biUnion f, M i j = #(A.biUnion f) * s := by
      simp [sum_comm (t := A.biUnion f), hM.2.2, mul_comm s]
    suffices #A * s ≤ #(A.biUnion f) * s by exact_mod_cast le_of_mul_le_mul_right this hs
    rw [← h₁, ← h₂]
    trans ∑ i ∈ A, ∑ j ∈ A.biUnion f, M i j
    · refine sum_le_sum fun i hi => ?_
      exact sum_le_sum_of_subset_of_nonneg (subset_biUnion_of_mem f hi) (by simp [*])
    · exact sum_le_sum_of_subset_of_nonneg (by simp) fun _ _ _ => sum_nonneg fun j _ => hM.1 _ _
  /-
    R : Type u_1
    n : Type u_2
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : LinearOrderedSemifield R
    M : Matrix n n R
    inst✝ : Nonempty n
    s : R
    hs : LT.lt 0 s
    hM : And (∀ (i j : n), LE.le 0 (M i j)) (And (∀ (i : n), Eq (Finset.univ.sum f …
    f : n → Finset n := fun i => Finset.filter (fun j => Ne (M i j) 0) Finset.univ
    hf : ∀ (A : Finset n), LE.le A.card (A.biUnion f).card
    ⊢ Exists fun σ => ∀ (i j : n), Eq (M i j) 0 → Eq (Equiv.Perm.permMatrix R σ i  …
  -/
  obtain ⟨g, hg, hg'⟩ := (all_card_le_biUnion_card_iff_exists_injective f).1 hf
  /-
    case intro.intro
    R : Type u_1
    n : Type u_2
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : LinearOrderedSemifield R
    M : Matrix n n R
    inst✝ : Nonempty n
    s : R
    hs : LT.lt 0 s
    hM : And (∀ (i j : n), LE.le 0 (M i j)) (And (∀ (i : n), Eq (Finset.univ.sum f …
    f : n → Finset n := fun i => Finset.filter (fun j => Ne (M i j) 0) Finset.univ
    hf : ∀ (A : Finset n), LE.le A.card (A.biUnion f).card
    g : n → n
    hg : Function.Injective g
    hg' : ∀ (x : n), Membership.mem (f x) (g x)
    ⊢ Exists fun σ => ∀ (i j : n), Eq (M i j) 0 → Eq (Equiv.Perm.permMatrix R σ i  …
  -/
  rw [Finite.injective_iff_bijective] at hg
  /-
    case intro.intro
    R : Type u_1
    n : Type u_2
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : LinearOrderedSemifield R
    M : Matrix n n R
    inst✝ : Nonempty n
    s : R
    hs : LT.lt 0 s
    hM : And (∀ (i j : n), LE.le 0 (M i j)) (And (∀ (i : n), Eq (Finset.univ.sum f …
    f : n → Finset n := fun i => Finset.filter (fun j => Ne (M i j) 0) Finset.univ
    hf : ∀ (A : Finset n), LE.le A.card (A.biUnion f).card
    g : n → n
    hg : Function.Bijective g
    hg' : ∀ (x : n), Membership.mem (f x) (g x)
    ⊢ Exists fun σ => ∀ (i j : n), Eq (M i j) 0 → Eq (Equiv.Perm.permMatrix R σ i  …
  -/
  refine ⟨Equiv.ofBijective g hg, fun i j hij => ?_⟩
  simp only [PEquiv.toMatrix_apply, Option.mem_def, ite_eq_right_iff, one_ne_zero, imp_false,
    Equiv.toPEquiv_apply, Equiv.ofBijective_apply, Option.some.injEq]
  /-
    case intro.intro
    R : Type u_1
    n : Type u_2
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : LinearOrderedSemifield R
    M : Matrix n n R
    inst✝ : Nonempty n
    s : R
    hs : LT.lt 0 s
    hM : And (∀ (i j : n), LE.le 0 (M i j)) (And (∀ (i : n), Eq (Finset.univ.sum f …
    f : n → Finset n := fun i => Finset.filter (fun j => Ne (M i j) 0) Finset.univ
    hf : ∀ (A : Finset n), LE.le A.card (A.biUnion f).card
    g : n → n
    hg : Function.Bijective g
    hg' : ∀ (x : n), Membership.mem (f x) (g x)
    i j : n
    hij : Eq (M i j) 0
    ⊢ Not (Eq (g i) j)
  -/
  rintro rfl
  /-
    case intro.intro
    R : Type u_1
    n : Type u_2
    inst✝³ : Fintype n
    inst✝² : DecidableEq n
    inst✝¹ : LinearOrderedSemifield R
    M : Matrix n n R
    inst✝ : Nonempty n
    s : R
    hs : LT.lt 0 s
    hM : And (∀ (i j : n), LE.le 0 (M i j)) (And (∀ (i : n), Eq (Finset.univ.sum f …
    f : n → Finset n := fun i => Finset.filter (fun j => Ne (M i j) 0) Finset.univ
    hf : ∀ (A : Finset n), LE.le A.card (A.biUnion f).card
    g : n → n
    hg : Function.Bijective g
    hg' : ∀ (x : n), Membership.mem (f x) (g x)
    i : n
    hij : Eq (M i (g i)) 0
    ⊢ False
  -/
  simpa [f, hij] using hg' i
  /-
    🎉 no goals
  -/


/--
If M is a scalar multiple of a doubly stochastic matrix, then it is a conical combination of
permutation matrices. This is most useful when M is a doubly stochastic matrix, in which case
the combination is convex.

This particular formulation is chosen to make the inductive step easier: we no longer need to
rescale each time a permutation matrix is subtracted.
-/
private lemma doublyStochastic_sum_perm_aux (M : Matrix n n R)
    (s : R) (hs : 0 ≤ s)
    (hM : ∃ M' ∈ doublyStochastic R n, M = s • M') :
    ∃ w : Equiv.Perm n → R, (∀ σ, 0 ≤ w σ) ∧ ∑ σ, w σ • σ.permMatrix R = M := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    s : R
    hs : LE.le 0 s
    hM : Exists fun M' => And (Membership.mem (doublyStochastic R n) M') (Eq M (HS …
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (Eq (Finset.univ.s …
  -/
  rcases isEmpty_or_nonempty n
  /-
    case inl
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    s : R
    hs : LE.le 0 s
    hM : Exists fun M' => And (Membership.mem (doublyStochastic R n) M') (Eq M (HS …
    h✝ : IsEmpty n
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (Eq (Finset.univ.s …
  -/
  case inl => exact ⟨1, by simp, Subsingleton.elim _ _⟩
  /-
    case inr
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    s : R
    hs : LE.le 0 s
    hM : Exists fun M' => And (Membership.mem (doublyStochastic R n) M') (Eq M (HS …
    h✝ : Nonempty n
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (Eq (Finset.univ.s …
  -/
  set d : ℕ := #{i : n × n | M i.1 i.2 ≠ 0} with ← hd
  /-
    case inr
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    s : R
    hs : LE.le 0 s
    hM : Exists fun M' => And (Membership.mem (doublyStochastic R n) M') (Eq M (HS …
    h✝ : Nonempty n
    d : Nat := (Finset.filter (fun i => Ne (M i.1 i.2) 0) Finset.univ).card
    hd : Eq (Finset.filter (fun i => Ne (M i.1 i.2) 0) Finset.univ).card d
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (Eq (Finset.univ.s …
  -/
  clear_value d
  /-
    case inr
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    s : R
    hs : LE.le 0 s
    hM : Exists fun M' => And (Membership.mem (doublyStochastic R n) M') (Eq M (HS …
    h✝ : Nonempty n
    d : Nat
    hd : Eq (Finset.filter (fun i => Ne (M i.1 i.2) 0) Finset.univ).card d
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (Eq (Finset.univ.s …
  -/
  induction d using Nat.strongRecOn generalizing M s
  case ind d ih =>
  rcases eq_or_lt_of_le hs with rfl | hs'
  case inl =>
    use 0
    simp only [zero_smul, exists_and_right] at hM
    simp [hM]
  obtain ⟨σ, hσ⟩ := exists_perm_eq_zero_implies_eq_zero hs' hM
  obtain ⟨i, hi, hi'⟩ := exists_min_image _ (fun i => M i (σ i)) univ_nonempty
  rw [exists_mem_doublyStochastic_eq_smul_iff hs] at hM
  let N : Matrix n n R := M - M i (σ i) • σ.permMatrix R
  have hMi' : 0 < M i (σ i) := (hM.1 _ _).lt_of_ne' fun h => by
    simpa [Equiv.toPEquiv_apply] using hσ _ _ h
  let s' : R := s - M i (σ i)
  have hs' : 0 ≤ s' := by
    simp only [s', sub_nonneg, ← hM.2.1 i]
    exact single_le_sum (fun j _ => hM.1 i j) (by simp)
  have : ∃ M' ∈ doublyStochastic R n, N = s' • M' := by
    rw [exists_mem_doublyStochastic_eq_smul_iff hs']
    simp only [sub_apply, smul_apply, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply, Option.mem_def,
      Option.some.injEq, smul_eq_mul, mul_ite, mul_one, mul_zero, sub_nonneg,
      sum_sub_distrib, sum_ite_eq, mem_univ, ↓reduceIte, N]
    refine ⟨fun i' j => ?_, by simp [s', hM.2.1], by simp [s', ← σ.eq_symm_apply, hM]⟩
    split
    case isTrue h => exact (hi' i' (by simp)).trans_eq (by rw [h])
    case isFalse h => exact hM.1 _ _
  have hd' : #{i : n × n | N i.1 i.2 ≠ 0} < d := by
    rw [← hd]
    refine card_lt_card ?_
    rw [ssubset_iff_of_subset (monotone_filter_right _ _)]
    · simp only [ne_eq, mem_filter, mem_univ, true_and, Decidable.not_not, Prod.exists]
      refine ⟨i, σ i, hMi'.ne', ?_⟩
      simp [N, Equiv.toPEquiv_apply]
    · rintro ⟨i', j'⟩ hN' hM'
      dsimp at hN' hM'
      simp only [sub_apply, hM', smul_apply, PEquiv.toMatrix_apply, Equiv.toPEquiv_apply,
        Option.mem_def, Option.some.injEq, smul_eq_mul, mul_ite, mul_one, mul_zero, zero_sub,
        neg_eq_zero, ite_eq_right_iff, Classical.not_imp, N] at hN'
      obtain ⟨rfl, _⟩ := hN'
      linarith [hi' i' (by simp)]
  obtain ⟨w, hw, hw'⟩ := ih _ hd' _ s' hs' this rfl
  refine ⟨w + fun σ' => if σ' = σ then M i (σ i) else 0, ?_⟩
  simp only [Pi.add_apply, add_smul, sum_add_distrib, hw', ite_smul, zero_smul,
    sum_ite_eq', mem_univ, ↓reduceIte, N, sub_add_cancel, and_true]
  intro σ'
  split <;> simp [add_nonneg, hw, hM.1]


/--
If M is a doubly stochastic matrix, then it is an convex combination of permutation matrices. Note
`doublyStochastic_eq_convexHull_permMatrix` shows `doublyStochastic n` is exactly the convex hull of
the permutation matrices, and this lemma is instead most useful for accessing the coefficients of
each permutation matrices directly.
-/
lemma exists_eq_sum_perm_of_mem_doublyStochastic (hM : M ∈ doublyStochastic R n) :
    ∃ w : Equiv.Perm n → R, (∀ σ, 0 ≤ w σ) ∧ ∑ σ, w σ = 1 ∧ ∑ σ, w σ • σ.permMatrix R = M := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    hM : Membership.mem (doublyStochastic R n) M
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (And (Eq (Finset.u …
  -/
  rcases isEmpty_or_nonempty n
  /-
    case inl
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    hM : Membership.mem (doublyStochastic R n) M
    h✝ : IsEmpty n
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (And (Eq (Finset.u …
  -/
  case inl => exact ⟨fun _ => 1, by simp, by simp, Subsingleton.elim _ _⟩
  /-
    case inr
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    hM : Membership.mem (doublyStochastic R n) M
    h✝ : Nonempty n
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (And (Eq (Finset.u …
  -/
  obtain ⟨w, hw1, hw3⟩ := doublyStochastic_sum_perm_aux M 1 (by simp) ⟨M, hM, by simp⟩
  /-
    case inr.intro.intro
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    hM : Membership.mem (doublyStochastic R n) M
    h✝ : Nonempty n
    w : Equiv.Perm n → R
    hw1 : ∀ (σ : Equiv.Perm n), LE.le 0 (w σ)
    hw3 : Eq (Finset.univ.sum fun σ => HSMul.hSMul (w σ) (Equiv.Perm.permMatrix R  …
    ⊢ Exists fun w => And (∀ (σ : Equiv.Perm n), LE.le 0 (w σ)) (And (Eq (Finset.u …
  -/
  refine ⟨w, hw1, ?_, hw3⟩
  /-
    case inr.intro.intro
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    hM : Membership.mem (doublyStochastic R n) M
    h✝ : Nonempty n
    w : Equiv.Perm n → R
    hw1 : ∀ (σ : Equiv.Perm n), LE.le 0 (w σ)
    hw3 : Eq (Finset.univ.sum fun σ => HSMul.hSMul (w σ) (Equiv.Perm.permMatrix R  …
    ⊢ Eq (Finset.univ.sum fun σ => w σ) 1
  -/
  inhabit n
  have : ∑ j, ∑ σ : Equiv.Perm n, w σ • σ.permMatrix R default j = 1 := by
    simp only [← smul_apply (m := n), ← Finset.sum_apply, hw3]
    rw [sum_row_of_mem_doublyStochastic hM]
  /-
    case inr.intro.intro
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    M : Matrix n n R
    hM : Membership.mem (doublyStochastic R n) M
    h✝ : Nonempty n
    w : Equiv.Perm n → R
    hw1 : ∀ (σ : Equiv.Perm n), LE.le 0 (w σ)
    hw3 : Eq (Finset.univ.sum fun σ => HSMul.hSMul (w σ) (Equiv.Perm.permMatrix R  …
    inhabited_h : Inhabited n
    this : Eq (Finset.univ.sum fun j => Finset.univ.sum fun σ => HSMul.hSMul (w σ) …
    ⊢ Eq (Finset.univ.sum fun σ => w σ) 1
  -/
  simpa [sum_comm (γ := n), Equiv.toPEquiv_apply] using this
  /-
    🎉 no goals
  -/


/--
**Birkhoff's theorem**
The set of doubly stochastic matrices is the convex hull of the permutation matrices.  Note
`exists_eq_sum_perm_of_mem_doublyStochastic` gives a convex weighting of each permutation matrix
directly.  To show `doublyStochastic n` is convex, use `convex_doublyStochastic`.
-/
theorem doublyStochastic_eq_convexHull_permMatrix :
    doublyStochastic R n = convexHull R {σ.permMatrix R | σ : Equiv.Perm n} := by
  /-
    R : Type u_1
    n : Type u_2
    inst✝² : Fintype n
    inst✝¹ : DecidableEq n
    inst✝ : LinearOrderedField R
    ⊢ Eq (↑(doublyStochastic R n)) ((convexHull R) (setOf fun x => Exists fun σ => …
  -/
  refine (convexHull_min ?g1 convex_doublyStochastic).antisymm' fun M hM => ?g2
  case g1 =>
    rintro x ⟨h, rfl⟩
    exact permMatrix_mem_doublyStochastic
  case g2 =>
    obtain ⟨w, hw1, hw2, hw3⟩ := exists_eq_sum_perm_of_mem_doublyStochastic hM
    exact mem_convexHull_of_exists_fintype w (·.permMatrix R) hw1 hw2 (by simp) hw3


