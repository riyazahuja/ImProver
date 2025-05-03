private def pairMap (t : Finset σ × σ) : Finset σ × σ :=
  if h : t.snd ∈ t.fst then (t.fst.erase t.snd, t.snd) else (t.fst.cons t.snd h, t.snd)


private lemma pairMap_ne_self (t : Finset σ × σ) : pairMap σ t ≠ t := by
  /-
    σ : Type u_1
    inst✝ : DecidableEq σ
    t : Prod (Finset σ) σ
    ⊢ Ne (MvPolynomial.NewtonIdentities.pairMap σ t) t
  -/
  rw [pairMap]
  /-
    σ : Type u_1
    inst✝ : DecidableEq σ
    t : Prod (Finset σ) σ
    ⊢ Ne (dite (Membership.mem t.1 t.2) (fun h => { fst := t.1.erase t.2, snd := t …
  -/
  split_ifs with h1
  /-
    case pos
    σ : Type u_1
    inst✝ : DecidableEq σ
    t : Prod (Finset σ) σ
    h1 : Membership.mem t.1 t.2
    ⊢ Ne { fst := t.1.erase t.2, snd := t.2 } t
  -/
  all_goals by_contra ht; rw [← ht] at h1; simp_all
  /-
    🎉 no goals
  -/


private lemma pairMap_of_snd_mem_fst {t : Finset σ × σ} (h : t.snd ∈ t.fst) :
    pairMap σ t = (t.fst.erase t.snd, t.snd) := by
  /-
    σ : Type u_1
    inst✝ : DecidableEq σ
    t : Prod (Finset σ) σ
    h : Membership.mem t.1 t.2
    ⊢ Eq (MvPolynomial.NewtonIdentities.pairMap σ t) { fst := t.1.erase t.2, snd : …
  -/
  simp [pairMap, h]
  /-
    🎉 no goals
  -/


private lemma pairMap_of_snd_nmem_fst {t : Finset σ × σ} (h : t.snd ∉ t.fst) :
    pairMap σ t = (t.fst.cons t.snd h, t.snd) := by
  /-
    σ : Type u_1
    inst✝ : DecidableEq σ
    t : Prod (Finset σ) σ
    h : Not (Membership.mem t.1 t.2)
    ⊢ Eq (MvPolynomial.NewtonIdentities.pairMap σ t) { fst := Finset.cons t.2 t.1  …
  -/
  simp [pairMap, h]
  /-
    🎉 no goals
  -/


@[simp]
private theorem pairMap_involutive : (pairMap σ).Involutive := by
  /-
    σ : Type u_1
    inst✝ : DecidableEq σ
    ⊢ Function.Involutive (MvPolynomial.NewtonIdentities.pairMap σ)
  -/
  intro t
  /-
    σ : Type u_1
    inst✝ : DecidableEq σ
    t : Prod (Finset σ) σ
    ⊢ Eq (MvPolynomial.NewtonIdentities.pairMap σ (MvPolynomial.NewtonIdentities.p …
  -/
  rw [pairMap, pairMap]
  /-
    σ : Type u_1
    inst✝ : DecidableEq σ
    t : Prod (Finset σ) σ
    ⊢ Eq (dite (Membership.mem (dite (Membership.mem t.1 t.2) (fun h => { fst := t …
  -/
  split_ifs with h1 h2 h3
    /-
      case pos
      σ : Type u_1
      inst✝ : DecidableEq σ
      t : Prod (Finset σ) σ
      h1 : Membership.mem t.1 t.2
      h2 : Membership.mem { fst := t.1.erase t.2, snd := t.2 }.1 { fst := t.1.erase  …
      ⊢ Eq { fst := { fst := t.1.erase t.2, snd := t.2 }.1.erase { fst := t.1.erase  …
    -/
  · simp at h2
    /-
      🎉 no goals
    -/
    /-
      case neg
      σ : Type u_1
      inst✝ : DecidableEq σ
      t : Prod (Finset σ) σ
      h1 : Membership.mem t.1 t.2
      h2 : Not (Membership.mem { fst := t.1.erase t.2, snd := t.2 }.1 { fst := t.1.e …
      ⊢ Eq { fst := Finset.cons { fst := t.1.erase t.2, snd := t.2 }.2 { fst := t.1. …
    -/
  · simp [insert_erase h1]
    /-
      🎉 no goals
    -/
    /-
      case pos
      σ : Type u_1
      inst✝ : DecidableEq σ
      t : Prod (Finset σ) σ
      h1 : Not (Membership.mem t.1 t.2)
      h3 : Membership.mem { fst := Finset.cons t.2 t.1 h1, snd := t.2 }.1 { fst := F …
      ⊢ Eq { fst := { fst := Finset.cons t.2 t.1 h1, snd := t.2 }.1.erase { fst := F …
    -/
  · simp_all
    /-
      🎉 no goals
    -/
    /-
      case neg
      σ : Type u_1
      inst✝ : DecidableEq σ
      t : Prod (Finset σ) σ
      h1 : Not (Membership.mem t.1 t.2)
      h3 : Not (Membership.mem { fst := Finset.cons t.2 t.1 h1, snd := t.2 }.1 { fst …
      ⊢ Eq { fst := Finset.cons { fst := Finset.cons t.2 t.1 h1, snd := t.2 }.2 { fs …
    -/
  · simp at h3
    /-
      🎉 no goals
    -/


private def pairs (k : ℕ) : Finset (Finset σ × σ) :=
  {t | #t.1 ≤ k ∧ (#t.1 = k → t.snd ∈ t.fst)}


@[simp]
private lemma mem_pairs (k : ℕ) (t : Finset σ × σ) :
    t ∈ pairs σ k ↔ #t.1 ≤ k ∧ (#t.1 = k → t.snd ∈ t.fst) := by
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    t : Prod (Finset σ) σ
    ⊢ Iff (Membership.mem (MvPolynomial.NewtonIdentities.pairs σ k) t) (And (LE.le …
  -/
  simp [pairs]
  /-
    🎉 no goals
  -/


private def weight (k : ℕ) (t : Finset σ × σ) : MvPolynomial σ R :=
  (-1) ^ #t.1 * ((∏ a ∈ t.fst, X a) * X t.snd ^ (k - #t.1))


private theorem pairMap_mem_pairs {k : ℕ} (t : Finset σ × σ) (h : t ∈ pairs σ k) :
    pairMap σ t ∈ pairs σ k := by
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    t : Prod (Finset σ) σ
    h : Membership.mem (MvPolynomial.NewtonIdentities.pairs σ k) t
    ⊢ Membership.mem (MvPolynomial.NewtonIdentities.pairs σ k) (MvPolynomial.Newto …
  -/
  rw [mem_pairs] at h ⊢
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    t : Prod (Finset σ) σ
    h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
    ⊢ And (LE.le (MvPolynomial.NewtonIdentities.pairMap σ t).1.card k) (Eq (MvPoly …
  -/
  rcases (em (t.snd ∈ t.fst)) with h1 | h1
    /-
      case inl
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h1 : Membership.mem t.1 t.2
      ⊢ And (LE.le (MvPolynomial.NewtonIdentities.pairMap σ t).1.card k) (Eq (MvPoly …
    -/
  · rw [pairMap_of_snd_mem_fst σ h1]
    /-
      case inl
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h1 : Membership.mem t.1 t.2
      ⊢ And (LE.le { fst := t.1.erase t.2, snd := t.2 }.1.card k) (Eq { fst := t.1.e …
    -/
    simp only [h1, implies_true, and_true] at h
    /-
      case inl
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h1 : Membership.mem t.1 t.2
      h : LE.le t.1.card k
      ⊢ And (LE.le { fst := t.1.erase t.2, snd := t.2 }.1.card k) (Eq { fst := t.1.e …
    -/
    simp only [card_erase_of_mem h1, tsub_le_iff_right, mem_erase, ne_eq, h1]
    /-
      case inl
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h1 : Membership.mem t.1 t.2
      h : LE.le t.1.card k
      ⊢ And (LE.le t.1.card (HAdd.hAdd k 1)) (Eq (HSub.hSub t.1.card 1) k → And (Not …
    -/
    refine ⟨le_step h, ?_⟩
    /-
      case inl
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h1 : Membership.mem t.1 t.2
      h : LE.le t.1.card k
      ⊢ Eq (HSub.hSub t.1.card 1) k → And (Not True) True
    -/
    by_contra h2
    /-
      case inl
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h1 : Membership.mem t.1 t.2
      h : LE.le t.1.card k
      h2 : Not (Eq (HSub.hSub t.1.card 1) k → And (Not True) True)
      ⊢ False
    -/
    simp only [not_true_eq_false, and_true, not_forall, not_false_eq_true, exists_prop] at h2
    /-
      case inl
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h1 : Membership.mem t.1 t.2
      h : LE.le t.1.card k
      h2 : Eq (HSub.hSub t.1.card 1) k
      ⊢ False
    -/
    rw [← h2] at h
    /-
      case inl
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h1 : Membership.mem t.1 t.2
      h : LE.le t.1.card (HSub.hSub t.1.card 1)
      h2 : Eq (HSub.hSub t.1.card 1) k
      ⊢ False
    -/
    exact not_le_of_lt (sub_lt (card_pos.mpr ⟨t.snd, h1⟩) zero_lt_one) h
    /-
      🎉 no goals
    -/
    /-
      case inr
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h1 : Not (Membership.mem t.1 t.2)
      ⊢ And (LE.le (MvPolynomial.NewtonIdentities.pairMap σ t).1.card k) (Eq (MvPoly …
    -/
  · rw [pairMap_of_snd_nmem_fst σ h1]
    /-
      case inr
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h1 : Not (Membership.mem t.1 t.2)
      ⊢ And (LE.le { fst := Finset.cons t.2 t.1 h1, snd := t.2 }.1.card k) (Eq { fst …
    -/
    simp only [h1] at h
    /-
      case inr
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h1 : Not (Membership.mem t.1 t.2)
      h : And (LE.le t.1.card k) (Eq t.1.card k → False)
      ⊢ And (LE.le { fst := Finset.cons t.2 t.1 h1, snd := t.2 }.1.card k) (Eq { fst …
    -/
    simp only [card_cons, mem_cons, true_or, implies_true, and_true]
    /-
      case inr
      σ : Type u_1
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h1 : Not (Membership.mem t.1 t.2)
      h : And (LE.le t.1.card k) (Eq t.1.card k → False)
      ⊢ LE.le (HAdd.hAdd t.1.card 1) k
    -/
    exact (le_iff_eq_or_lt.mp h.left).resolve_left h.right
    /-
      🎉 no goals
    -/


private theorem weight_add_weight_pairMap {k : ℕ} (t : Finset σ × σ) (h : t ∈ pairs σ k) :
    weight σ R k t + weight σ R k (pairMap σ t) = 0 := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    t : Prod (Finset σ) σ
    h : Membership.mem (MvPolynomial.NewtonIdentities.pairs σ k) t
    ⊢ Eq (HAdd.hAdd (MvPolynomial.NewtonIdentities.weight σ R k t) (MvPolynomial.N …
  -/
  rw [weight, weight]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    t : Prod (Finset σ) σ
    h : Membership.mem (MvPolynomial.NewtonIdentities.pairs σ k) t
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul (t.1.prod fun  …
  -/
  rw [mem_pairs] at h
  have h2 (n : ℕ) : -(-1 : MvPolynomial σ R) ^ n = (-1) ^ (n + 1) := by
    rw [← neg_one_mul ((-1 : MvPolynomial σ R) ^ n), pow_add, pow_one, mul_comm]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    t : Prod (Finset σ) σ
    h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
    h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
    ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul (t.1.prod fun  …
  -/
  rcases (em (t.snd ∈ t.fst)) with h1 | h1
    /-
      case inl
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Membership.mem t.1 t.2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul (t.1.prod fun  …
    -/
  · rw [pairMap_of_snd_mem_fst σ h1]
    simp only [← prod_erase_mul t.fst (fun j ↦ (X j : MvPolynomial σ R)) h1,
      mul_assoc (∏ a ∈ erase t.fst t.snd, X a), card_erase_of_mem h1]
    /-
      case inl
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Membership.mem t.1 t.2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul ((t.1.erase t. …
    -/
    nth_rewrite 1 [← pow_one (X t.snd)]
    /-
      case inl
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Membership.mem t.1 t.2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul ((t.1.erase t. …
    -/
    simp only [← pow_add, add_comm]
    /-
      case inl
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Membership.mem t.1 t.2
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul ((t.1.erase t. …
    -/
    have h3 : 1 ≤ #t.1 := lt_iff_add_one_le.mp (card_pos.mpr ⟨t.snd, h1⟩)
    rw [← tsub_tsub_assoc h.left h3, ← neg_neg ((-1 : MvPolynomial σ R) ^ (#t.1 - 1)),
      h2 (#t.1 - 1), Nat.sub_add_cancel h3]
    /-
      case inl
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Membership.mem t.1 t.2
      h3 : LE.le 1 t.1.card
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul ((t.1.erase t. …
    -/
    simp
    /-
      🎉 no goals
    -/
    /-
      case inr
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Not (Membership.mem t.1 t.2)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul (t.1.prod fun  …
    -/
  · rw [pairMap_of_snd_nmem_fst σ h1]
    /-
      case inr
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Not (Membership.mem t.1 t.2)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (HPow.hPow (-1) t.1.card) (HMul.hMul (t.1.prod fun  …
    -/
    simp only [mul_comm, mul_assoc (∏ a ∈ t.fst, X a), card_cons, prod_cons]
    /-
      case inr
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Not (Membership.mem t.1 t.2)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (t.1.prod fun a => MvPolynomial.X a) (HMul.hMul (HP …
    -/
    nth_rewrite 2 [← pow_one (X t.snd)]
    simp only [← pow_add, ← Nat.add_sub_assoc (Nat.lt_of_le_of_ne h.left (mt h.right h1)), add_comm,
      Nat.succ_eq_add_one, Nat.add_sub_add_right]
    /-
      case inr
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Not (Membership.mem t.1 t.2)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (t.1.prod fun a => MvPolynomial.X a) (HMul.hMul (HP …
    -/
    rw [← neg_neg ((-1 : MvPolynomial σ R) ^ #t.1), h2]
    /-
      case inr
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      t : Prod (Finset σ) σ
      h : And (LE.le t.1.card k) (Eq t.1.card k → Membership.mem t.1 t.2)
      h2 : ∀ (n : Nat), Eq (Neg.neg (HPow.hPow (-1) n)) (HPow.hPow (-1) (HAdd.hAdd n …
      h1 : Not (Membership.mem t.1 t.2)
      ⊢ Eq (HAdd.hAdd (HMul.hMul (t.1.prod fun a => MvPolynomial.X a) (HMul.hMul (HP …
    -/
    simp
    /-
      🎉 no goals
    -/


private theorem weight_sum (k : ℕ) : ∑ t ∈ pairs σ k, weight σ R k t = 0 :=
  sum_involution (fun t _ ↦ pairMap σ t) (weight_add_weight_pairMap σ R)
    (fun t _ ↦ (fun _ ↦ pairMap_ne_self σ t)) (pairMap_mem_pairs σ)
    (fun t _ ↦ pairMap_involutive σ t)


private theorem sum_filter_pairs_eq_sum_powersetCard_sum (k : ℕ)
    (f : Finset σ × σ → MvPolynomial σ R) :
    ∑ t ∈ pairs σ k with #t.1 = k, f t = ∑ A ∈ powersetCard k univ, ∑ j ∈ A, f (A, j) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    f : Prod (Finset σ) σ → MvPolynomial σ R
    ⊢ Eq ((Finset.filter (fun t => Eq t.1.card k) (MvPolynomial.NewtonIdentities.p …
  -/
  apply sum_finset_product
  /-
    case h
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    f : Prod (Finset σ) σ → MvPolynomial σ R
    ⊢ ∀ (p : Prod (Finset σ) σ), Iff (Membership.mem (Finset.filter (fun t => Eq t …
  -/
  aesop
  /-
    🎉 no goals
  -/


private theorem sum_filter_pairs_eq_sum_powersetCard_mem_filter_antidiagonal_sum (k : ℕ) (a : ℕ × ℕ)
    (ha : a ∈ {a ∈ antidiagonal k | a.fst < k}) (f : Finset σ × σ → MvPolynomial σ R) :
    ∑ t ∈ pairs σ k with #t.1 = a.1, f t = ∑ A ∈ powersetCard a.1 univ, ∑ j, f (A, j) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.filter (fun a => LT.lt a.1 k) (Finset.HasAntidiago …
    f : Prod (Finset σ) σ → MvPolynomial σ R
    ⊢ Eq ((Finset.filter (fun t => Eq t.1.card a.1) (MvPolynomial.NewtonIdentities …
  -/
  apply sum_finset_product
  /-
    case h
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.filter (fun a => LT.lt a.1 k) (Finset.HasAntidiago …
    f : Prod (Finset σ) σ → MvPolynomial σ R
    ⊢ ∀ (p : Prod (Finset σ) σ), Iff (Membership.mem (Finset.filter (fun t => Eq t …
  -/
  simp only [mem_filter, mem_powersetCard_univ, mem_univ, and_true, and_iff_right_iff_imp]
  /-
    case h
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.filter (fun a => LT.lt a.1 k) (Finset.HasAntidiago …
    f : Prod (Finset σ) σ → MvPolynomial σ R
    ⊢ ∀ (p : Prod (Finset σ) σ), Eq p.1.card a.1 → Membership.mem (MvPolynomial.Ne …
  -/
  rintro p hp
  /-
    case h
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.filter (fun a => LT.lt a.1 k) (Finset.HasAntidiago …
    f : Prod (Finset σ) σ → MvPolynomial σ R
    p : Prod (Finset σ) σ
    hp : Eq p.1.card a.1
    ⊢ Membership.mem (MvPolynomial.NewtonIdentities.pairs σ k) p
  -/
  have : #p.fst ≤ k := by apply le_of_lt; aesop
  /-
    case h
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.filter (fun a => LT.lt a.1 k) (Finset.HasAntidiago …
    f : Prod (Finset σ) σ → MvPolynomial σ R
    p : Prod (Finset σ) σ
    hp : Eq p.1.card a.1
    this : LE.le p.1.card k
    ⊢ Membership.mem (MvPolynomial.NewtonIdentities.pairs σ k) p
  -/
  aesop
  /-
    🎉 no goals
  -/


private lemma filter_pairs_lt (k : ℕ) :
    (pairs σ k).filter (fun (s, _) ↦ #s < k) =
      (range k).disjiUnion (powersetCard · univ) ((pairwise_disjoint_powersetCard _).set_pairwise _)
                      /-
                        σ : Type u_1
                        inst✝¹ : DecidableEq σ
                        inst✝ : Fintype σ
                        k : Nat
                        ⊢ Eq (Finset.filter (fun x => MvPolynomial.NewtonIdentities.filter_pairs_lt.ma …
                      -/
        ×ˢ univ := by ext; aesop (add unsafe le_of_lt)
                           /-
                             🎉 no goals
                           -/


private theorem sum_filter_pairs_eq_sum_filter_antidiagonal_powersetCard_sum (k : ℕ)
    (f : Finset σ × σ → MvPolynomial σ R) :
    ∑ t ∈ pairs σ k with #t.1 < k, f t =
      ∑ a ∈ antidiagonal k with a.fst < k, ∑ A ∈ powersetCard a.fst univ, ∑ j, f (A, j) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    f : Prod (Finset σ) σ → MvPolynomial σ R
    ⊢ Eq ((Finset.filter (fun t => LT.lt t.1.card k) (MvPolynomial.NewtonIdentitie …
  -/
  rw [filter_pairs_lt, sum_product, sum_disjiUnion]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝² : CommRing R
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    f : Prod (Finset σ) σ → MvPolynomial σ R
    ⊢ Eq ((Finset.range k).sum fun i => (Finset.powersetCard i Finset.univ).sum fu …
  -/
  refine sum_nbij' (fun n ↦ (n, k - n)) Prod.fst ?_ ?_ ?_ ?_ ?_ <;>
    /-
      case refine_1
      σ : Type u_1
      R : Type u_2
      inst✝² : CommRing R
      inst✝¹ : DecidableEq σ
      inst✝ : Fintype σ
      k : Nat
      f : Prod (Finset σ) σ → MvPolynomial σ R
      ⊢ ∀ (a : Nat), Membership.mem (Finset.range k) a → Membership.mem (Finset.filt …
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
    /-
      🎉 no goals
    -/
    simp +contextual [@eq_comm _ _ k, Nat.add_sub_cancel', le_of_lt]
    /-
      🎉 no goals
    -/


private theorem disjoint_filter_pairs_lt_filter_pairs_eq (k : ℕ) :
    Disjoint {t ∈ pairs σ k | #t.1 < k} {t ∈ pairs σ k | #t.1 = k} := by
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    ⊢ Disjoint (Finset.filter (fun t => LT.lt t.1.card k) (MvPolynomial.NewtonIden …
  -/
  rw [disjoint_filter]
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    ⊢ ∀ (x : Prod (Finset σ) σ), Membership.mem (MvPolynomial.NewtonIdentities.pai …
  -/
  exact fun _ _ h1 h2 ↦ lt_irrefl _ (h2.symm.subst h1)
  /-
    🎉 no goals
  -/


private theorem disjUnion_filter_pairs_eq_pairs (k : ℕ) :
    disjUnion {t ∈ pairs σ k | #t.1 < k} {t ∈ pairs σ k | #t.1 = k}
      (disjoint_filter_pairs_lt_filter_pairs_eq σ k) = pairs σ k := by
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    ⊢ Eq ((Finset.filter (fun t => LT.lt t.1.card k) (MvPolynomial.NewtonIdentitie …
  -/
  simp only [disjUnion_eq_union, Finset.ext_iff, pairs, filter_filter, mem_filter]
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    ⊢ ∀ (a : Prod (Finset σ) σ), Iff (Membership.mem (Union.union (Finset.filter ( …
  -/
  intro a
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod (Finset σ) σ
    ⊢ Iff (Membership.mem (Union.union (Finset.filter (fun a => And (And (LE.le a. …
  -/
  rw [← filter_or, mem_filter]
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod (Finset σ) σ
    ⊢ Iff (And (Membership.mem Finset.univ a) (Or (And (And (LE.le a.1.card k) (Eq …
  -/
  refine ⟨fun ha ↦ by tauto, fun ha ↦ ?_⟩
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod (Finset σ) σ
    ha : And (Membership.mem Finset.univ a) (And (LE.le a.1.card k) (Eq a.1.card k …
    ⊢ And (Membership.mem Finset.univ a) (Or (And (And (LE.le a.1.card k) (Eq a.1. …
  -/
  have hacard := le_iff_lt_or_eq.mp ha.2.1
  /-
    σ : Type u_1
    inst✝¹ : DecidableEq σ
    inst✝ : Fintype σ
    k : Nat
    a : Prod (Finset σ) σ
    ha : And (Membership.mem Finset.univ a) (And (LE.le a.1.card k) (Eq a.1.card k …
    hacard : Or (LT.lt a.1.card k) (Eq a.1.card k)
    ⊢ And (Membership.mem Finset.univ a) (Or (And (And (LE.le a.1.card k) (Eq a.1. …
  -/
  tauto
  /-
    🎉 no goals
  -/


private theorem esymm_summand_to_weight (k : ℕ) (A : Finset σ) (h : A ∈ powersetCard k univ) :
    ∑ j ∈ A, weight σ R k (A, j) = k * (-1) ^ k * (∏ i ∈ A, X i : MvPolynomial σ R) := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    k : Nat
    A : Finset σ
    h : Membership.mem (Finset.powersetCard k Finset.univ) A
    ⊢ Eq (A.sum fun j => MvPolynomial.NewtonIdentities.weight σ R k { fst := A, sn …
  -/
  simp [weight, mem_powersetCard_univ.mp h, mul_assoc]
  /-
    🎉 no goals
  -/


private theorem esymm_to_weight [DecidableEq σ] (k : ℕ) : k * esymm σ R k =
    (-1) ^ k * ∑ t ∈ pairs σ k with #t.1 = k, weight σ R k t := by
  rw [esymm, sum_filter_pairs_eq_sum_powersetCard_sum σ R k (fun t ↦ weight σ R k t),
    sum_congr rfl (esymm_summand_to_weight σ R k), mul_comm (k : MvPolynomial σ R) ((-1) ^ k),
    ← mul_sum, ← mul_assoc, ← mul_assoc, ← pow_add, Even.neg_one_pow ⟨k, rfl⟩, one_mul]


private theorem esymm_mul_psum_summand_to_weight (k : ℕ) (a : ℕ × ℕ) (ha : a ∈ antidiagonal k) :
    ∑ A ∈ powersetCard a.fst univ, ∑ j, weight σ R k (A, j) =
    (-1) ^ a.fst * esymm σ R a.fst * psum σ R a.snd := by
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) a
    ⊢ Eq ((Finset.powersetCard a.1 Finset.univ).sum fun A => Finset.univ.sum fun j …
  -/
  simp only [esymm, psum, weight, ← mul_assoc, mul_sum]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) a
    ⊢ Eq ((Finset.powersetCard a.1 Finset.univ).sum fun x => Finset.univ.sum fun x …
  -/
  rw [sum_comm]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) a
    ⊢ Eq (Finset.univ.sum fun y => (Finset.powersetCard a.1 Finset.univ).sum fun x …
  -/
  refine sum_congr rfl fun x _ ↦ ?_
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) a
    x : σ
    x✝ : Membership.mem Finset.univ x
    ⊢ Eq ((Finset.powersetCard a.1 Finset.univ).sum fun x_1 => HMul.hMul (HMul.hMu …
  -/
  rw [sum_mul]
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) a
    x : σ
    x✝ : Membership.mem Finset.univ x
    ⊢ Eq ((Finset.powersetCard a.1 Finset.univ).sum fun x_1 => HMul.hMul (HMul.hMu …
  -/
  refine sum_congr rfl fun s hs ↦ ?_
  /-
    σ : Type u_1
    R : Type u_2
    inst✝¹ : CommRing R
    inst✝ : Fintype σ
    k : Nat
    a : Prod Nat Nat
    ha : Membership.mem (Finset.HasAntidiagonal.antidiagonal k) a
    x : σ
    x✝ : Membership.mem Finset.univ x
    s : Finset σ
    hs : Membership.mem (Finset.powersetCard a.1 Finset.univ) s
    ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow (-1) s.card) (s.prod fun a => MvPolynomi …
  -/
  rw [mem_powersetCard_univ.mp hs, ← mem_antidiagonal.mp ha, add_sub_self_left]
  /-
    🎉 no goals
  -/


private theorem esymm_mul_psum_to_weight [DecidableEq σ] (k : ℕ) :
    ∑ a ∈ antidiagonal k with a.fst < k, (-1) ^ a.fst * esymm σ R a.fst * psum σ R a.snd =
      ∑ t ∈ pairs σ k with #t.1 < k, weight σ R k t := by
  rw [← sum_congr rfl (fun a ha ↦ esymm_mul_psum_summand_to_weight σ R k a (mem_filter.mp ha).left),
    sum_filter_pairs_eq_sum_filter_antidiagonal_powersetCard_sum σ R k]


/-- **Newton's identities** give a recurrence relation for the kth elementary symmetric polynomial
in terms of lower degree elementary symmetric polynomials and power sums. -/
theorem mul_esymm_eq_sum (k : ℕ) :
    k * esymm σ R k = (-1) ^ (k + 1) *
      ∑ a ∈ antidiagonal k with a.1 < k, (-1) ^ a.1 * esymm σ R a.1 * psum σ R a.2 := by
  classical
  rw [NewtonIdentities.esymm_to_weight σ R k, NewtonIdentities.esymm_mul_psum_to_weight σ R k,
    eq_comm, ← sub_eq_zero, sub_eq_add_neg, neg_mul_eq_neg_mul,
    neg_eq_neg_one_mul ((-1 : MvPolynomial σ R) ^ k)]
  nth_rw 2 [← pow_one (-1 : MvPolynomial σ R)]
  rw [← pow_add, add_comm 1 k, ← left_distrib,
    ← sum_disjUnion (NewtonIdentities.disjoint_filter_pairs_lt_filter_pairs_eq σ k),
    NewtonIdentities.disjUnion_filter_pairs_eq_pairs σ k, NewtonIdentities.weight_sum σ R k,
    neg_one_pow_mul_eq_zero_iff.mpr rfl]


theorem sum_antidiagonal_card_esymm_psum_eq_zero :
    ∑ a ∈ antidiagonal (Fintype.card σ), (-1) ^ a.fst * esymm σ R a.fst * psum σ R a.snd = 0 := by
  /-
    σ : Type u_1
    inst✝¹ : Fintype σ
    R : Type u_2
    inst✝ : CommRing R
    ⊢ Eq ((Finset.HasAntidiagonal.antidiagonal (Fintype.card σ)).sum fun a => HMul …
  -/
  let k := Fintype.card σ
  suffices (-1 : MvPolynomial σ R) ^ (k + 1) *
      ∑ a ∈ antidiagonal k, (-1) ^ a.fst * esymm σ R a.fst * psum σ R a.snd = 0 by
    simpa using this
  simp [k, ← sum_filter_add_sum_filter_not (antidiagonal k) (fun a ↦ a.fst < k),
    ← mul_esymm_eq_sum, mul_add, ← mul_assoc, ← pow_add, mul_comm ↑k (esymm σ R k)]


/-- A version of Newton's identities which may be more useful in the case that we know the values of
the elementary symmetric polynomials and would like to calculate the values of the power sums. -/
theorem psum_eq_mul_esymm_sub_sum (k : ℕ) (h : 0 < k) :
    psum σ R k = (-1) ^ (k + 1) * k * esymm σ R k -
    ∑ a ∈ antidiagonal k with a.1 ∈ Set.Ioo 0 k, (-1) ^ a.fst * esymm σ R a.1 * psum σ R a.2 := by
  /-
    σ : Type u_1
    inst✝¹ : Fintype σ
    R : Type u_2
    inst✝ : CommRing R
    k : Nat
    h : LT.lt 0 k
    ⊢ Eq (MvPolynomial.psum σ R k) (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1 …
  -/
  simp only [Set.Ioo, Set.mem_setOf_eq, and_comm]
  /-
    σ : Type u_1
    inst✝¹ : Fintype σ
    R : Type u_2
    inst✝ : CommRing R
    k : Nat
    h : LT.lt 0 k
    ⊢ Eq (MvPolynomial.psum σ R k) (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1 …
  -/
  have hesymm := mul_esymm_eq_sum σ R k
  rw [← (sum_filter_add_sum_filter_not {a ∈ antidiagonal k | a.fst < k}
    (fun a ↦ 0 < a.fst) (fun a ↦ (-1) ^ a.fst * esymm σ R a.fst * psum σ R a.snd))] at hesymm
  have sub_both_sides := congrArg (· - (-1 : MvPolynomial σ R) ^ (k + 1) *
    ∑ a ∈ {a ∈ antidiagonal k | a.fst < k} with 0 < a.fst,
    (-1) ^ a.fst * esymm σ R a.fst * psum σ R a.snd) hesymm
  /-
    σ : Type u_1
    inst✝¹ : Fintype σ
    R : Type u_2
    inst✝ : CommRing R
    k : Nat
    h : LT.lt 0 k
    hesymm : Eq (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (HMul.hMul (HPow.hPow  …
    sub_both_sides : Eq ((fun x => HSub.hSub x (HMul.hMul (HPow.hPow (-1) (HAdd.hA …
    ⊢ Eq (MvPolynomial.psum σ R k) (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1 …
  -/
  simp only [left_distrib, add_sub_cancel_left] at sub_both_sides
  /-
    σ : Type u_1
    inst✝¹ : Fintype σ
    R : Type u_2
    inst✝ : CommRing R
    k : Nat
    h : LT.lt 0 k
    hesymm : Eq (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (HMul.hMul (HPow.hPow  …
    sub_both_sides : Eq (HSub.hSub (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (HM …
    ⊢ Eq (MvPolynomial.psum σ R k) (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1 …
  -/
  have sub_both_sides := congrArg ((-1 : MvPolynomial σ R) ^ (k + 1) * ·) sub_both_sides
  simp only [mul_sub_left_distrib, ← mul_assoc, ← pow_add, Even.neg_one_pow ⟨k + 1, rfl⟩, one_mul,
    not_le, lt_one_iff, filter_filter (fun a : ℕ × ℕ ↦ a.fst < k) (fun a ↦ ¬0 < a.fst)]
    at sub_both_sides
  have : {a ∈ antidiagonal k | a.fst < k ∧ ¬0 < a.fst} = {(0, k)} := by
    ext a
    rw [mem_filter, mem_antidiagonal, mem_singleton]
    refine ⟨?_, by rintro rfl; omega⟩
    rintro ⟨ha, ⟨_, ha0⟩⟩
    rw [← ha, Nat.eq_zero_of_not_pos ha0, zero_add, ← Nat.eq_zero_of_not_pos ha0]
  /-
    σ : Type u_1
    inst✝¹ : Fintype σ
    R : Type u_2
    inst✝ : CommRing R
    k : Nat
    h : LT.lt 0 k
    hesymm : Eq (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (HMul.hMul (HPow.hPow  …
    sub_both_sides✝ : Eq (HSub.hSub (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (H …
    sub_both_sides : Eq (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAd …
    this : Eq (Finset.filter (fun a => And (LT.lt a.1 k) (Not (LT.lt 0 a.1))) (Fin …
    ⊢ Eq (MvPolynomial.psum σ R k) (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1 …
  -/
  rw [this, sum_singleton] at sub_both_sides
  /-
    σ : Type u_1
    inst✝¹ : Fintype σ
    R : Type u_2
    inst✝ : CommRing R
    k : Nat
    h : LT.lt 0 k
    hesymm : Eq (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (HMul.hMul (HPow.hPow  …
    sub_both_sides✝ : Eq (HSub.hSub (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (H …
    sub_both_sides : Eq (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAd …
    this : Eq (Finset.filter (fun a => And (LT.lt a.1 k) (Not (LT.lt 0 a.1))) (Fin …
    ⊢ Eq (MvPolynomial.psum σ R k) (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1 …
  -/
  simp only [_root_.pow_zero, esymm_zero, mul_one, one_mul, filter_filter] at sub_both_sides
  /-
    σ : Type u_1
    inst✝¹ : Fintype σ
    R : Type u_2
    inst✝ : CommRing R
    k : Nat
    h : LT.lt 0 k
    hesymm : Eq (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (HMul.hMul (HPow.hPow  …
    sub_both_sides✝ : Eq (HSub.hSub (HMul.hMul (↑k) (MvPolynomial.esymm σ R k)) (H …
    this : Eq (Finset.filter (fun a => And (LT.lt a.1 k) (Not (LT.lt 0 a.1))) (Fin …
    sub_both_sides : Eq (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1) (HAdd.hAd …
    ⊢ Eq (MvPolynomial.psum σ R k) (HSub.hSub (HMul.hMul (HMul.hMul (HPow.hPow (-1 …
  -/
  exact sub_both_sides.symm
  /-
    🎉 no goals
  -/


