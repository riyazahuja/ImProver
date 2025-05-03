/-- The `j`th entry of `accumulate n m t` is the sum of `t i` over all `i ≥ j`. -/
@[simps] def accumulate (n m : ℕ) : (Fin n → ℕ) →+ (Fin m → ℕ) where
  toFun t j := ∑ i : Fin n with j.val ≤ i.val, t i
  map_zero' := funext <| fun _ ↦ sum_eq_zero <| fun _ _ ↦ rfl
                                         /-
                                           σ : Type u_1
                                           τ : Type u_2
                                           R : Type u_3
                                           n✝ m✝ k n m : Nat
                                           t₁ t₂ : Fin n → Nat
                                           j : Fin m
                                           ⊢ Eq ({ toFun := fun t j => (Finset.filter (fun i => LE.le ↑j ↑i) Finset.univ) …
                                         -/
  map_add' t₁ t₂ := funext <| fun j ↦ by dsimp only; exact sum_add_distrib
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- The `i`th entry of `invAccumulate n m s` is `s i - s (i+1)`, where `s j = 0` if `j ≥ m`. -/
def invAccumulate (n m : ℕ) (s : Fin m → ℕ) (i : Fin n) : ℕ :=
  (if hi : i < m then s ⟨i, hi⟩ else 0) - (if hi : i + 1 < m then s ⟨i + 1, hi⟩ else 0)


lemma accumulate_rec {i n m : ℕ} (hin : i < n) (him : i + 1 < m) (t : Fin n → ℕ) :
    accumulate n m t ⟨i, Nat.lt_of_succ_lt him⟩ = t ⟨i, hin⟩ + accumulate n m t ⟨i + 1, him⟩ := by
  /-
    i n m : Nat
    hin : LT.lt i n
    him : LT.lt (HAdd.hAdd i 1) m
    t : Fin n → Nat
    ⊢ Eq ((Fin.accumulate n m) t ⟨i, ⋯⟩) (HAdd.hAdd (t ⟨i, hin⟩) ((Fin.accumulate  …
  -/
  simp_rw [accumulate_apply]
  /-
    i n m : Nat
    hin : LT.lt i n
    him : LT.lt (HAdd.hAdd i 1) m
    t : Fin n → Nat
    ⊢ Eq ((Finset.filter (fun i_1 => LE.le i ↑i_1) Finset.univ).sum fun i => t i)  …
  -/
  convert (add_sum_erase _ _ _).symm
    /-
      case h.e'_3.h.e'_6.h
      i n m : Nat
      hin : LT.lt i n
      him : LT.lt (HAdd.hAdd i 1) m
      t : Fin n → Nat
      ⊢ Eq (Finset.filter (fun i_1 => LE.le (HAdd.hAdd i 1) ↑i_1) Finset.univ) ((Fin …
    -/
  · ext
    /-
      case h.e'_3.h.e'_6.h.h
      i n m : Nat
      hin : LT.lt i n
      him : LT.lt (HAdd.hAdd i 1) m
      t : Fin n → Nat
      a✝ : Fin n
      ⊢ Iff (Membership.mem (Finset.filter (fun i_1 => LE.le (HAdd.hAdd i 1) ↑i_1) F …
    -/
    rw [mem_erase]
    /-
      case h.e'_3.h.e'_6.h.h
      i n m : Nat
      hin : LT.lt i n
      him : LT.lt (HAdd.hAdd i 1) m
      t : Fin n → Nat
      a✝ : Fin n
      ⊢ Iff (Membership.mem (Finset.filter (fun i_1 => LE.le (HAdd.hAdd i 1) ↑i_1) F …
    -/
    simp_rw [mem_filter, mem_univ, true_and, i.succ_le_iff, lt_iff_le_and_ne]
    /-
      case h.e'_3.h.e'_6.h.h
      i n m : Nat
      hin : LT.lt i n
      him : LT.lt (HAdd.hAdd i 1) m
      t : Fin n → Nat
      a✝ : Fin n
      ⊢ Iff (And (LE.le i ↑a✝) (Ne i ↑a✝)) (And (Ne a✝ ⟨i, hin⟩) (LE.le i ↑a✝))
    -/
    rw [and_comm, ne_comm, ← Fin.val_ne_iff]
    /-
      🎉 no goals
    -/
    /-
      case convert_7
      i n m : Nat
      hin : LT.lt i n
      him : LT.lt (HAdd.hAdd i 1) m
      t : Fin n → Nat
      ⊢ Membership.mem (Finset.filter (fun i_1 => LE.le i ↑i_1) Finset.univ) ⟨i, hin⟩
    -/
  · exact mem_filter.2 ⟨mem_univ _, le_rfl⟩
    /-
      🎉 no goals
    -/


lemma accumulate_last {i n m : ℕ} (hin : i < n) (hmi : m = i + 1) (t : Fin n → ℕ)
    (ht : ∀ j : Fin n, m ≤ j → t j = 0) :
    accumulate n m t ⟨i, i.lt_succ_self.trans_eq hmi.symm⟩ = t ⟨i, hin⟩ := by
  /-
    i n m : Nat
    hin : LT.lt i n
    hmi : Eq m (HAdd.hAdd i 1)
    t : Fin n → Nat
    ht : ∀ (j : Fin n), LE.le m ↑j → Eq (t j) 0
    ⊢ Eq ((Fin.accumulate n m) t ⟨i, ⋯⟩) (t ⟨i, hin⟩)
  -/
  rw [accumulate_apply]
  /-
    i n m : Nat
    hin : LT.lt i n
    hmi : Eq m (HAdd.hAdd i 1)
    t : Fin n → Nat
    ht : ∀ (j : Fin n), LE.le m ↑j → Eq (t j) 0
    ⊢ Eq ((Finset.filter (fun i_1 => LE.le ↑⟨i, ⋯⟩ ↑i_1) Finset.univ).sum fun i => …
  -/
  apply sum_eq_single_of_mem
    /-
      case h
      i n m : Nat
      hin : LT.lt i n
      hmi : Eq m (HAdd.hAdd i 1)
      t : Fin n → Nat
      ht : ∀ (j : Fin n), LE.le m ↑j → Eq (t j) 0
      ⊢ Membership.mem (Finset.filter (fun i_1 => LE.le ↑⟨i, ⋯⟩ ↑i_1) Finset.univ) ⟨ …
    -/
  · rw [mem_filter]; exact ⟨mem_univ _, le_rfl⟩
                     /-
                       🎉 no goals
                     -/
  /-
    case h₀
    i n m : Nat
    hin : LT.lt i n
    hmi : Eq m (HAdd.hAdd i 1)
    t : Fin n → Nat
    ht : ∀ (j : Fin n), LE.le m ↑j → Eq (t j) 0
    ⊢ ∀ (b : Fin n), Membership.mem (Finset.filter (fun i_1 => LE.le ↑⟨i, ⋯⟩ ↑i_1) …
  -/
  refine fun j hij hji ↦ ht j ?_
  /-
    case h₀
    i n m : Nat
    hin : LT.lt i n
    hmi : Eq m (HAdd.hAdd i 1)
    t : Fin n → Nat
    ht : ∀ (j : Fin n), LE.le m ↑j → Eq (t j) 0
    j : Fin n
    hij : Membership.mem (Finset.filter (fun i_1 => LE.le ↑⟨i, ⋯⟩ ↑i_1) Finset.uni …
    hji : Ne j ⟨i, hin⟩
    ⊢ LE.le m ↑j
  -/
  simp_rw [mem_filter, mem_univ, true_and] at hij
  /-
    case h₀
    i n m : Nat
    hin : LT.lt i n
    hmi : Eq m (HAdd.hAdd i 1)
    t : Fin n → Nat
    ht : ∀ (j : Fin n), LE.le m ↑j → Eq (t j) 0
    j : Fin n
    hji : Ne j ⟨i, hin⟩
    hij : LE.le i ↑j
    ⊢ LE.le m ↑j
  -/
  exact hmi.trans_le (hij.lt_of_ne (Fin.val_ne_iff.2 hji).symm).nat_succ_le
  /-
    🎉 no goals
  -/


lemma accumulate_injective {n m} (hnm : n ≤ m) : Function.Injective (accumulate n m) := by
  /-
    n m : Nat
    hnm : LE.le n m
    ⊢ Function.Injective ⇑(Fin.accumulate n m)
  -/
  refine fun t s he ↦ funext fun i ↦ ?_
  /-
    n m : Nat
    hnm : LE.le n m
    t s : Fin n → Nat
    he : Eq ((Fin.accumulate n m) t) ((Fin.accumulate n m) s)
    i : Fin n
    ⊢ Eq (t i) (s i)
  -/
  obtain h|h := lt_or_le (i.1 + 1) m
    /-
      case inl
      n m : Nat
      hnm : LE.le n m
      t s : Fin n → Nat
      he : Eq ((Fin.accumulate n m) t) ((Fin.accumulate n m) s)
      i : Fin n
      h : LT.lt (HAdd.hAdd (↑i) 1) m
      ⊢ Eq (t i) (s i)
    -/
  · have := accumulate_rec i.2 h s
    /-
      case inl
      n m : Nat
      hnm : LE.le n m
      t s : Fin n → Nat
      he : Eq ((Fin.accumulate n m) t) ((Fin.accumulate n m) s)
      i : Fin n
      h : LT.lt (HAdd.hAdd (↑i) 1) m
      this : Eq ((Fin.accumulate n m) s ⟨↑i, ⋯⟩) (HAdd.hAdd (s ⟨↑i, ⋯⟩) ((Fin.accumu …
      ⊢ Eq (t i) (s i)
    -/
    rwa [← he, accumulate_rec i.2 h t, add_right_cancel_iff] at this
    /-
      🎉 no goals
    -/
    /-
      case inr
      n m : Nat
      hnm : LE.le n m
      t s : Fin n → Nat
      he : Eq ((Fin.accumulate n m) t) ((Fin.accumulate n m) s)
      i : Fin n
      h : LE.le m (HAdd.hAdd (↑i) 1)
      ⊢ Eq (t i) (s i)
    -/
  · have := h.antisymm (i.2.nat_succ_le.trans hnm)
    /-
      case inr
      n m : Nat
      hnm : LE.le n m
      t s : Fin n → Nat
      he : Eq ((Fin.accumulate n m) t) ((Fin.accumulate n m) s)
      i : Fin n
      h : LE.le m (HAdd.hAdd (↑i) 1)
      this : Eq m (HAdd.hAdd (↑i) 1)
      ⊢ Eq (t i) (s i)
    -/
    rw [← accumulate_last i.2 this t, ← accumulate_last i.2 this s, he]
    /-
      case inr
      n m : Nat
      hnm : LE.le n m
      t s : Fin n → Nat
      he : Eq ((Fin.accumulate n m) t) ((Fin.accumulate n m) s)
      i : Fin n
      h : LE.le m (HAdd.hAdd (↑i) 1)
      this : Eq m (HAdd.hAdd (↑i) 1)
      ⊢ ∀ (j : Fin n), LE.le m ↑j → Eq (s j) 0
    -/
    iterate 2 { intro j hj; exact ((j.2.trans_le hnm).not_le hj).elim }
    /-
      🎉 no goals
    -/


lemma accumulate_invAccumulate {n m} (hmn : m ≤ n) {s : Fin m → ℕ} (hs : Antitone s) :
    accumulate n m (invAccumulate n m s) = s := funext <| fun ⟨i, hi⟩ ↦ by
  /-
    n m : Nat
    hmn : LE.le m n
    s : Fin m → Nat
    hs : Antitone s
    x✝ : Fin m
    i : Nat
    hi : LT.lt i m
    ⊢ Eq ((Fin.accumulate n m) (Fin.invAccumulate n m s) ⟨i, hi⟩) (s ⟨i, hi⟩)
  -/
  have := Nat.le_sub_one_of_lt hi
  /-
    n m : Nat
    hmn : LE.le m n
    s : Fin m → Nat
    hs : Antitone s
    x✝ : Fin m
    i : Nat
    hi : LT.lt i m
    this : LE.le i (HSub.hSub m 1)
    ⊢ Eq ((Fin.accumulate n m) (Fin.invAccumulate n m s) ⟨i, hi⟩) (s ⟨i, hi⟩)
  -/
  revert hi
  /-
    n m : Nat
    hmn : LE.le m n
    s : Fin m → Nat
    hs : Antitone s
    x✝ : Fin m
    i : Nat
    this : LE.le i (HSub.hSub m 1)
    ⊢ ∀ (hi : LT.lt i m), Eq ((Fin.accumulate n m) (Fin.invAccumulate n m s) ⟨i, h …
  -/
  refine Nat.decreasingInduction' (fun i hi _ ih him ↦ ?_) this fun hm ↦ ?_
    /-
      case refine_1
      n m : Nat
      hmn : LE.le m n
      s : Fin m → Nat
      hs : Antitone s
      x✝¹ : Fin m
      i✝ : Nat
      this : LE.le i✝ (HSub.hSub m 1)
      i : Nat
      hi : LT.lt i (HSub.hSub m 1)
      x✝ : LE.le i✝ i
      ih : ∀ (hi : LT.lt (HAdd.hAdd i 1) m), Eq ((Fin.accumulate n m) (Fin.invAccumu …
      him : LT.lt i m
      ⊢ Eq ((Fin.accumulate n m) (Fin.invAccumulate n m s) ⟨i, him⟩) (s ⟨i, him⟩)
    -/
  · rw [← Nat.pred_eq_sub_one, Nat.lt_pred_iff, Nat.succ_eq_add_one] at hi
    /-
      case refine_1
      n m : Nat
      hmn : LE.le m n
      s : Fin m → Nat
      hs : Antitone s
      x✝¹ : Fin m
      i✝ : Nat
      this : LE.le i✝ (HSub.hSub m 1)
      i : Nat
      hi : LT.lt (HAdd.hAdd i 1) m
      x✝ : LE.le i✝ i
      ih : ∀ (hi : LT.lt (HAdd.hAdd i 1) m), Eq ((Fin.accumulate n m) (Fin.invAccumu …
      him : LT.lt i m
      ⊢ Eq ((Fin.accumulate n m) (Fin.invAccumulate n m s) ⟨i, him⟩) (s ⟨i, him⟩)
    -/
    rw [accumulate_rec (him.trans_le hmn) hi, ih hi, invAccumulate, dif_pos him, dif_pos hi]
    /-
      case refine_1
      n m : Nat
      hmn : LE.le m n
      s : Fin m → Nat
      hs : Antitone s
      x✝¹ : Fin m
      i✝ : Nat
      this : LE.le i✝ (HSub.hSub m 1)
      i : Nat
      hi : LT.lt (HAdd.hAdd i 1) m
      x✝ : LE.le i✝ i
      ih : ∀ (hi : LT.lt (HAdd.hAdd i 1) m), Eq ((Fin.accumulate n m) (Fin.invAccumu …
      him : LT.lt i m
      ⊢ Eq (HAdd.hAdd (HSub.hSub (s ⟨↑⟨i, ⋯⟩, him⟩) (s ⟨HAdd.hAdd (↑⟨i, ⋯⟩) 1, hi⟩)) …
    -/
    simp only
    /-
      case refine_1
      n m : Nat
      hmn : LE.le m n
      s : Fin m → Nat
      hs : Antitone s
      x✝¹ : Fin m
      i✝ : Nat
      this : LE.le i✝ (HSub.hSub m 1)
      i : Nat
      hi : LT.lt (HAdd.hAdd i 1) m
      x✝ : LE.le i✝ i
      ih : ∀ (hi : LT.lt (HAdd.hAdd i 1) m), Eq ((Fin.accumulate n m) (Fin.invAccumu …
      him : LT.lt i m
      ⊢ Eq (HAdd.hAdd (HSub.hSub (s ⟨i, him⟩) (s ⟨HAdd.hAdd i 1, hi⟩)) (s ⟨HAdd.hAdd …
    -/
    exact Nat.sub_add_cancel (hs i.le_succ)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n m : Nat
      hmn : LE.le m n
      s : Fin m → Nat
      hs : Antitone s
      x✝ : Fin m
      i : Nat
      this : LE.le i (HSub.hSub m 1)
      hm : LT.lt (HSub.hSub m 1) m
      ⊢ Eq ((Fin.accumulate n m) (Fin.invAccumulate n m s) ⟨HSub.hSub m 1, hm⟩) (s ⟨ …
    -/
  · have := (Nat.sub_one_add_one <| Nat.not_eq_zero_of_lt hm).symm
    rw [accumulate_last (hm.trans_le hmn) this, invAccumulate, dif_pos hm, dif_neg this.not_gt,
      Nat.sub_zero]
    /-
      case refine_2.ht
      n m : Nat
      hmn : LE.le m n
      s : Fin m → Nat
      hs : Antitone s
      x✝ : Fin m
      i : Nat
      this✝ : LE.le i (HSub.hSub m 1)
      hm : LT.lt (HSub.hSub m 1) m
      this : Eq m (HAdd.hAdd (HSub.hSub m 1) 1)
      ⊢ ∀ (j : Fin n), LE.le m ↑j → Eq (Fin.invAccumulate n m s j) 0
    -/
    intro j hj
    /-
      case refine_2.ht
      n m : Nat
      hmn : LE.le m n
      s : Fin m → Nat
      hs : Antitone s
      x✝ : Fin m
      i : Nat
      this✝ : LE.le i (HSub.hSub m 1)
      hm : LT.lt (HSub.hSub m 1) m
      this : Eq m (HAdd.hAdd (HSub.hSub m 1) 1)
      j : Fin n
      hj : LE.le m ↑j
      ⊢ Eq (Fin.invAccumulate n m s j) 0
    -/
    rw [invAccumulate, dif_neg hj.not_lt, Nat.zero_sub]
    /-
      🎉 no goals
    -/


variable (σ R n) in
/-- The `R`-algebra homomorphism from $R[x_1,\dots,x_n]$ to the symmetric subalgebra of
  $R[\{x_i \mid i ∈ σ\}]$ sending $x_i$ to the $i$-th elementary symmetric polynomial. -/
noncomputable def esymmAlgHom :
    MvPolynomial (Fin n) R →ₐ[R] symmetricSubalgebra σ R :=
  aeval (fun i ↦ ⟨esymm σ R (i + 1), esymm_isSymmetric σ R _⟩)


lemma esymmAlgHom_apply (p : MvPolynomial (Fin n) R) :
    (esymmAlgHom σ R n p).val = aeval (fun i : Fin n ↦ esymm σ R (i + 1)) p :=
  (Subalgebra.mvPolynomial_aeval_coe _ _ _).symm


lemma rename_esymmAlgHom (e : σ ≃ τ) :
    (renameSymmetricSubalgebra e).toAlgHom.comp (esymmAlgHom σ R n) = esymmAlgHom τ R n := by
  /-
    σ : Type u_1
    τ : Type u_2
    R : Type u_3
    n : Nat
    inst✝² : CommSemiring R
    inst✝¹ : Fintype σ
    inst✝ : Fintype τ
    e : Equiv σ τ
    ⊢ Eq ((↑(MvPolynomial.renameSymmetricSubalgebra e)).comp (MvPolynomial.esymmAl …
  -/
  ext i : 2
  simp_rw [AlgHom.comp_apply, esymmAlgHom, aeval_X, AlgEquiv.toAlgHom_eq_coe, AlgHom.coe_coe,
    renameSymmetricSubalgebra_apply_coe, rename_esymm]


variable (σ) in
/-- The image of a monomial under `esymmAlgHom`. -/
noncomputable def esymmAlgHomMonomial (t : Fin n →₀ ℕ) (r : R) :
    MvPolynomial σ R := (esymmAlgHom σ R n <| monomial t r).val


lemma isSymmetric_esymmAlgHomMonomial (t : Fin n →₀ ℕ) (r : R) :
    (esymmAlgHomMonomial σ t r).IsSymmetric := (esymmAlgHom _ _ _ _).2


lemma esymmAlgHomMonomial_single :
    esymmAlgHomMonomial σ (Finsupp.single i k) r = C r * esymm σ R (i + 1) ^ k := by
  rw [esymmAlgHomMonomial, esymmAlgHom_apply, aeval_monomial, algebraMap_eq,
    Finsupp.prod_single_index]
  /-
    σ : Type u_1
    R : Type u_3
    n k : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    i : Fin n
    r : R
    ⊢ Eq (HPow.hPow (MvPolynomial.esymm σ R (HAdd.hAdd (↑i) 1)) 0) 1
  -/
  exact pow_zero _
  /-
    🎉 no goals
  -/


lemma esymmAlgHomMonomial_single_one :
    esymmAlgHomMonomial σ (Finsupp.single i k) 1 = esymm σ R (i + 1) ^ k := by
  /-
    σ : Type u_1
    R : Type u_3
    n k : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    i : Fin n
    ⊢ Eq (MvPolynomial.esymmAlgHomMonomial σ (Finsupp.single i k) 1) (HPow.hPow (M …
  -/
  rw [esymmAlgHomMonomial_single, map_one, one_mul]
  /-
    🎉 no goals
  -/


lemma esymmAlgHomMonomial_add {t s : Fin n →₀ ℕ} :
    esymmAlgHomMonomial σ (t + s) r = esymmAlgHomMonomial σ t r * esymmAlgHomMonomial σ s 1 := by
  /-
    σ : Type u_1
    R : Type u_3
    n : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    r : R
    t s : Finsupp (Fin n) Nat
    ⊢ Eq (MvPolynomial.esymmAlgHomMonomial σ (HAdd.hAdd t s) r) (HMul.hMul (MvPoly …
  -/
  simp_rw [esymmAlgHomMonomial, esymmAlgHom_apply, ← map_mul, monomial_mul, mul_one]
  /-
    🎉 no goals
  -/


lemma esymmAlgHom_zero : esymmAlgHomMonomial σ (0 : Fin n →₀ ℕ) r = C r := by
  /-
    σ : Type u_1
    R : Type u_3
    n : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Fintype σ
    r : R
    ⊢ Eq (MvPolynomial.esymmAlgHomMonomial σ 0 r) (MvPolynomial.C r)
  -/
  rw [esymmAlgHomMonomial, monomial_zero', esymmAlgHom_apply, aeval_C, algebraMap_eq]
  /-
    🎉 no goals
  -/


private lemma supDegree_monic_esymm [Nontrivial R] {i : ℕ} (him : i < m) :
    supDegree toLex (esymm (Fin m) R (i + 1)) =
      toLex (Finsupp.indicator (Iic ⟨i, him⟩) fun _ _ ↦ 1) ∧
    Monic toLex (esymm (Fin m) R (i + 1)) := by
  have := supDegree_leadingCoeff_sum_eq (D := toLex) (s := univ.powersetCard (i + 1))
    (i := Iic (⟨i, him⟩ : Fin m)) ?_ (f := fun s ↦ monomial (∑ j in s, fun₀ | j => 1) (1 : R)) ?_
  · rwa [← esymm_eq_sum_monomial, ← Finsupp.indicator_eq_sum_single, ← single_eq_monomial,
      supDegree_single_ne_zero _ one_ne_zero, leadingCoeff_single toLex.injective] at this
    /-
      case refine_1
      R : Type u_3
      m : Nat
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      i : Nat
      him : LT.lt i m
      ⊢ Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) (Finset.Iic …
    -/
  · exact mem_powersetCard.2 ⟨subset_univ _, Fin.card_Iic _⟩
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_3
    m : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i : Nat
    him : LT.lt i m
    ⊢ ∀ (j : Finset (Fin m)), Membership.mem (Finset.powersetCard (HAdd.hAdd i 1)  …
  -/
  intro t ht hne
  have ht' : #t = #(Iic (⟨i, him⟩ : Fin m)) := by
    rw [(mem_powersetCard.1 ht).2, Fin.card_Iic]
  simp_rw [← single_eq_monomial, supDegree_single_ne_zero _ one_ne_zero,
    ← Finsupp.indicator_eq_sum_single]
  /-
    case refine_2
    R : Type u_3
    m : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i : Nat
    him : LT.lt i m
    t : Finset (Fin m)
    ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
    hne : Ne t (Finset.Iic ⟨i, him⟩)
    ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
    ⊢ LT.lt (toLex (Finsupp.indicator t fun x x => 1)) (toLex (Finsupp.indicator ( …
  -/
  rw [ne_comm, Ne, ← subset_iff_eq_of_card_le ht'.le, not_subset] at hne
  /-
    case refine_2
    R : Type u_3
    m : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i : Nat
    him : LT.lt i m
    t : Finset (Fin m)
    ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
    hne : Exists fun x => And (Membership.mem (Finset.Iic ⟨i, him⟩) x) (Not (Membe …
    ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
    ⊢ LT.lt (toLex (Finsupp.indicator t fun x x => 1)) (toLex (Finsupp.indicator ( …
  -/
  simp_rw [← mem_sdiff] at hne
  /-
    case refine_2
    R : Type u_3
    m : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i : Nat
    him : LT.lt i m
    t : Finset (Fin m)
    ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
    ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
    hne : Exists fun x => Membership.mem (SDiff.sdiff (Finset.Iic ⟨i, him⟩) t) x
    ⊢ LT.lt (toLex (Finsupp.indicator t fun x x => 1)) (toLex (Finsupp.indicator ( …
  -/
  have hkm := mem_sdiff.1 (min'_mem _ hne)
  /-
    case refine_2
    R : Type u_3
    m : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i : Nat
    him : LT.lt i m
    t : Finset (Fin m)
    ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
    ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
    hne : Exists fun x => Membership.mem (SDiff.sdiff (Finset.Iic ⟨i, him⟩) t) x
    hkm : And (Membership.mem (Finset.Iic ⟨i, him⟩) ((SDiff.sdiff (Finset.Iic ⟨i,  …
    ⊢ LT.lt (toLex (Finsupp.indicator t fun x x => 1)) (toLex (Finsupp.indicator ( …
  -/
  refine ⟨min' _ hne, fun k hk ↦ ?_, ?_⟩
  /-
    case refine_2.refine_1
    R : Type u_3
    m : Nat
    inst✝¹ : CommSemiring R
    inst✝ : Nontrivial R
    i : Nat
    him : LT.lt i m
    t : Finset (Fin m)
    ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
    ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
    hne : Exists fun x => Membership.mem (SDiff.sdiff (Finset.Iic ⟨i, him⟩) t) x
    hkm : And (Membership.mem (Finset.Iic ⟨i, him⟩) ((SDiff.sdiff (Finset.Iic ⟨i,  …
    k : Fin m
    hk : (fun x1 x2 => LT.lt x1 x2) k ((SDiff.sdiff (Finset.Iic ⟨i, him⟩) t).min'  …
    ⊢ Eq ((ofLex (toLex (Finsupp.indicator t fun x x => 1))) k) ((ofLex (toLex (Fi …
  -/
  all_goals simp only [Pi.toLex_apply, ofLex_toLex, Finsupp.indicator_apply]
    /-
      case refine_2.refine_1
      R : Type u_3
      m : Nat
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      i : Nat
      him : LT.lt i m
      t : Finset (Fin m)
      ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
      ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
      hne : Exists fun x => Membership.mem (SDiff.sdiff (Finset.Iic ⟨i, him⟩) t) x
      hkm : And (Membership.mem (Finset.Iic ⟨i, him⟩) ((SDiff.sdiff (Finset.Iic ⟨i,  …
      k : Fin m
      hk : (fun x1 x2 => LT.lt x1 x2) k ((SDiff.sdiff (Finset.Iic ⟨i, him⟩) t).min'  …
      ⊢ Eq (dite (Membership.mem t k) (fun hi => 1) fun hi => 0) (dite (Membership.m …
    -/
  · have hki := mem_Iic.2 (hk.le.trans <| mem_Iic.1 hkm.1)
    /-
      case refine_2.refine_1
      R : Type u_3
      m : Nat
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      i : Nat
      him : LT.lt i m
      t : Finset (Fin m)
      ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
      ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
      hne : Exists fun x => Membership.mem (SDiff.sdiff (Finset.Iic ⟨i, him⟩) t) x
      hkm : And (Membership.mem (Finset.Iic ⟨i, him⟩) ((SDiff.sdiff (Finset.Iic ⟨i,  …
      k : Fin m
      hk : (fun x1 x2 => LT.lt x1 x2) k ((SDiff.sdiff (Finset.Iic ⟨i, him⟩) t).min'  …
      hki : Membership.mem (Finset.Iic ⟨i, him⟩) k
      ⊢ Eq (dite (Membership.mem t k) (fun hi => 1) fun hi => 0) (dite (Membership.m …
    -/
    rw [dif_pos hki, dif_pos]
    /-
      case refine_2.refine_1.hc
      R : Type u_3
      m : Nat
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      i : Nat
      him : LT.lt i m
      t : Finset (Fin m)
      ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
      ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
      hne : Exists fun x => Membership.mem (SDiff.sdiff (Finset.Iic ⟨i, him⟩) t) x
      hkm : And (Membership.mem (Finset.Iic ⟨i, him⟩) ((SDiff.sdiff (Finset.Iic ⟨i,  …
      k : Fin m
      hk : (fun x1 x2 => LT.lt x1 x2) k ((SDiff.sdiff (Finset.Iic ⟨i, him⟩) t).min'  …
      hki : Membership.mem (Finset.Iic ⟨i, him⟩) k
      ⊢ Membership.mem t k
    -/
    by_contra h
    /-
      case refine_2.refine_1.hc
      R : Type u_3
      m : Nat
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      i : Nat
      him : LT.lt i m
      t : Finset (Fin m)
      ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
      ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
      hne : Exists fun x => Membership.mem (SDiff.sdiff (Finset.Iic ⟨i, him⟩) t) x
      hkm : And (Membership.mem (Finset.Iic ⟨i, him⟩) ((SDiff.sdiff (Finset.Iic ⟨i,  …
      k : Fin m
      hk : (fun x1 x2 => LT.lt x1 x2) k ((SDiff.sdiff (Finset.Iic ⟨i, him⟩) t).min'  …
      hki : Membership.mem (Finset.Iic ⟨i, him⟩) k
      h : Not (Membership.mem t k)
      ⊢ False
    -/
    exact lt_irrefl k <| ((lt_min'_iff _ _).1 hk) _ <| mem_sdiff.2 ⟨hki, h⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2.refine_2
      R : Type u_3
      m : Nat
      inst✝¹ : CommSemiring R
      inst✝ : Nontrivial R
      i : Nat
      him : LT.lt i m
      t : Finset (Fin m)
      ht : Membership.mem (Finset.powersetCard (HAdd.hAdd i 1) Finset.univ) t
      ht' : Eq t.card (Finset.Iic ⟨i, him⟩).card
      hne : Exists fun x => Membership.mem (SDiff.sdiff (Finset.Iic ⟨i, him⟩) t) x
      hkm : And (Membership.mem (Finset.Iic ⟨i, him⟩) ((SDiff.sdiff (Finset.Iic ⟨i,  …
      ⊢ LT.lt (dite (Membership.mem t ((SDiff.sdiff (Finset.Iic ⟨i, him⟩) t).min' hn …
    -/
  · rw [dif_neg hkm.2, dif_pos hkm.1]; exact Nat.zero_lt_one
                                       /-
                                         🎉 no goals
                                       -/


lemma supDegree_esymm [Nontrivial R] (him : i < m) :
    ofLex (supDegree toLex <| esymm (Fin m) R (i + 1)) = accumulate n m (Finsupp.single i 1) := by
  /-
    R : Type u_3
    n m : Nat
    inst✝¹ : CommSemiring R
    i : Fin n
    inst✝ : Nontrivial R
    him : LT.lt (↑i) m
    ⊢ Eq (⇑(ofLex (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymm (Fin m) …
  -/
  rw [(supDegree_monic_esymm him).1, ofLex_toLex]
  /-
    R : Type u_3
    n m : Nat
    inst✝¹ : CommSemiring R
    i : Fin n
    inst✝ : Nontrivial R
    him : LT.lt (↑i) m
    ⊢ Eq (⇑(Finsupp.indicator (Finset.Iic ⟨↑i, him⟩) fun x x => 1)) ((Fin.accumula …
  -/
  ext j
  simp_rw [Finsupp.indicator_apply, dite_eq_ite, mem_Iic, accumulate_apply, Finsupp.single_apply,
    sum_ite_eq, mem_filter, mem_univ, true_and, Fin.le_def]


lemma monic_esymm {i : ℕ} (him : i ≤ m) : Monic toLex (esymm (Fin m) R i) := by
  cases i with
  | zero =>
    rw [esymm_zero]
    exact monic_one toLex.injective
  | succ i =>
    nontriviality R
    exact (supDegree_monic_esymm him).2


lemma leadingCoeff_esymmAlgHomMonomial (t : Fin n →₀ ℕ) (hnm : n ≤ m) :
    leadingCoeff toLex (esymmAlgHomMonomial (Fin m) t r) = r := by
  induction t using Finsupp.induction₂ with
  | h0 => rw [esymmAlgHom_zero, leadingCoeff_toLex_C]
  | ha i _ _ _ _ ih =>
    rw [esymmAlgHomMonomial_add, esymmAlgHomMonomial_single_one,
        ((monic_esymm <| i.2.trans_le hnm).pow toLex_add toLex.injective).leadingCoeff_mul_eq_left,
        ih]
    exacts [toLex.injective, toLex_add]


lemma supDegree_esymmAlgHomMonomial (hr : r ≠ 0) (t : Fin n →₀ ℕ) (hnm : n ≤ m) :
    ofLex (supDegree toLex <| esymmAlgHomMonomial (Fin m) t r) = accumulate n m t := by
  /-
    R : Type u_3
    n m : Nat
    inst✝ : CommSemiring R
    r : R
    hr : Ne r 0
    t : Finsupp (Fin n) Nat
    hnm : LE.le n m
    ⊢ Eq (⇑(ofLex (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMo …
  -/
  nontriviality R
  induction t using Finsupp.induction₂ with
  | h0 => simp_rw [esymmAlgHom_zero, supDegree_toLex_C, ofLex_zero, Finsupp.coe_zero, map_zero]
  | ha  i _ _ _ _ ih =>
    have := i.2.trans_le hnm
    rw [esymmAlgHomMonomial_add, esymmAlgHomMonomial_single_one,
        Monic.supDegree_mul_of_ne_zero_left toLex.injective toLex_add, ofLex_add, Finsupp.coe_add,
        ih, Finsupp.coe_add, map_add, Monic.supDegree_pow rfl toLex_add toLex.injective, ofLex_smul,
        Finsupp.coe_smul, supDegree_esymm this, ← map_nsmul, ← Finsupp.coe_smul,
        Finsupp.smul_single, nsmul_one, Nat.cast_id]
    · exact monic_esymm this
    · exact (monic_esymm this).pow toLex_add toLex.injective
    · rwa [Ne, ← leadingCoeff_eq_zero toLex.injective, leadingCoeff_esymmAlgHomMonomial _ hnm]


omit [Fintype σ] in
lemma IsSymmetric.antitone_supDegree [LinearOrder σ] {p : MvPolynomial σ R} (hp : p.IsSymmetric) :
    Antitone ↑(ofLex <| p.supDegree toLex) := by
  /-
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : LinearOrder σ
    p : MvPolynomial σ R
    hp : p.IsSymmetric
    ⊢ Antitone ⇑(ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p))
  -/
  obtain rfl | h0 := eq_or_ne p 0
    /-
      case inl
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      hp : MvPolynomial.IsSymmetric 0
      ⊢ Antitone ⇑(ofLex (AddMonoidAlgebra.supDegree (⇑toLex) 0))
    -/
  · rw [supDegree_zero, Finsupp.bot_eq_zero]
    /-
      case inl
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      hp : MvPolynomial.IsSymmetric 0
      ⊢ Antitone ⇑(ofLex 0)
    -/
    exact Pi.zero_mono
    /-
      🎉 no goals
    -/
  /-
    case inr
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : LinearOrder σ
    p : MvPolynomial σ R
    hp : p.IsSymmetric
    h0 : Ne p 0
    ⊢ Antitone ⇑(ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p))
  -/
  rw [Antitone]
  /-
    case inr
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : LinearOrder σ
    p : MvPolynomial σ R
    hp : p.IsSymmetric
    h0 : Ne p 0
    ⊢ ∀ ⦃a b : σ⦄, LE.le a b → LE.le ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex)  …
  -/
  by_contra! h
  /-
    case inr
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : LinearOrder σ
    p : MvPolynomial σ R
    hp : p.IsSymmetric
    h0 : Ne p 0
    h : Exists fun ⦃a⦄ => Exists fun ⦃b⦄ => And (LE.le a b) (LT.lt ((ofLex (AddMon …
    ⊢ False
  -/
  obtain ⟨i, j, hle, hlt⟩ := h
  /-
    case inr.intro.intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : LinearOrder σ
    p : MvPolynomial σ R
    hp : p.IsSymmetric
    h0 : Ne p 0
    i j : σ
    hle : LE.le i j
    hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
    ⊢ False
  -/
  apply (le_sup (s := p.support) (f := toLex) _).not_lt
  /-
    case inr.intro.intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : LinearOrder σ
    p : MvPolynomial σ R
    hp : p.IsSymmetric
    h0 : Ne p 0
    i j : σ
    hle : LE.le i j
    hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
    ⊢ LT.lt (p.support.sup ⇑toLex) (toLex ?m.79303)
  -/
  pick_goal 3
    /-
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      p : MvPolynomial σ R
      hp : p.IsSymmetric
      h0 : Ne p 0
      i j : σ
      hle : LE.le i j
      hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
      ⊢ Membership.mem p.support ?m.79303
    -/
  · rw [← hp (Equiv.swap i j), mem_support_iff, coeff_rename_mapDomain _ (Equiv.injective _)]
    /-
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      p : MvPolynomial σ R
      hp : p.IsSymmetric
      h0 : Ne p 0
      i j : σ
      hle : LE.le i j
      hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
      ⊢ Ne (MvPolynomial.coeff ?d p) 0
    -/
    rw [Ne, ← leadingCoeff_eq_zero toLex.injective, leadingCoeff_toLex] at h0
    /-
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      p : MvPolynomial σ R
      hp : p.IsSymmetric
      h0✝ : Ne p 0
      h0 : Not (Eq (MvPolynomial.coeff (ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p …
      i j : σ
      hle : LE.le i j
      hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
      ⊢ Ne (MvPolynomial.coeff (?m.79822 i j hle hlt) p) 0
    -/
    assumption
    /-
      🎉 no goals
    -/
  /-
    case inr.intro.intro.intro
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : LinearOrder σ
    p : MvPolynomial σ R
    hp : p.IsSymmetric
    h0 : Ne p 0
    i j : σ
    hle : LE.le i j
    hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
    ⊢ LT.lt (p.support.sup ⇑toLex) (toLex (Finsupp.mapDomain (⇑(Equiv.swap i j)) ( …
  -/
  refine ⟨i, fun k hk ↦ ?_, ?_⟩
  /-
    case inr.intro.intro.intro.refine_1
    σ : Type u_1
    R : Type u_3
    inst✝¹ : CommSemiring R
    inst✝ : LinearOrder σ
    p : MvPolynomial σ R
    hp : p.IsSymmetric
    h0 : Ne p 0
    i j : σ
    hle : LE.le i j
    hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
    k : σ
    hk : (fun x1 x2 => LT.lt x1 x2) k i
    ⊢ Eq ((ofLex (p.support.sup ⇑toLex)) k) ((ofLex (toLex (Finsupp.mapDomain (⇑(E …
  -/
  all_goals dsimp only [Pi.toLex_apply, ofLex_toLex]
    /-
      case inr.intro.intro.intro.refine_1
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      p : MvPolynomial σ R
      hp : p.IsSymmetric
      h0 : Ne p 0
      i j : σ
      hle : LE.le i j
      hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
      k : σ
      hk : (fun x1 x2 => LT.lt x1 x2) k i
      ⊢ Eq ((ofLex (p.support.sup ⇑toLex)) k) ((Finsupp.mapDomain (⇑(Equiv.swap i j) …
    -/
  · conv_rhs => rw [← Equiv.swap_apply_of_ne_of_ne hk.ne (hk.trans_le hle).ne]
    /-
      case inr.intro.intro.intro.refine_1
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      p : MvPolynomial σ R
      hp : p.IsSymmetric
      h0 : Ne p 0
      i j : σ
      hle : LE.le i j
      hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
      k : σ
      hk : (fun x1 x2 => LT.lt x1 x2) k i
      ⊢ Eq ((ofLex (p.support.sup ⇑toLex)) k) ((Finsupp.mapDomain (⇑(Equiv.swap i j) …
    -/
    rw [Finsupp.mapDomain_apply (Equiv.injective _), supDegree]; rfl
                                                                 /-
                                                                   🎉 no goals
                                                                 -/
    /-
      case inr.intro.intro.intro.refine_2
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      p : MvPolynomial σ R
      hp : p.IsSymmetric
      h0 : Ne p 0
      i j : σ
      hle : LE.le i j
      hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
      ⊢ LT.lt ((ofLex (p.support.sup ⇑toLex)) i) ((Finsupp.mapDomain (⇑(Equiv.swap i …
    -/
  · apply hlt.trans_eq
    /-
      case inr.intro.intro.intro.refine_2
      σ : Type u_1
      R : Type u_3
      inst✝¹ : CommSemiring R
      inst✝ : LinearOrder σ
      p : MvPolynomial σ R
      hp : p.IsSymmetric
      h0 : Ne p 0
      i j : σ
      hle : LE.le i j
      hlt : LT.lt ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) i) ((ofLex (AddMo …
      ⊢ Eq ((ofLex (AddMonoidAlgebra.supDegree (⇑toLex) p)) j) ((Finsupp.mapDomain ( …
    -/
    simp_rw [Finsupp.mapDomain_equiv_apply, Equiv.symm_swap, Equiv.swap_apply_left]
    /-
      🎉 no goals
    -/


lemma esymmAlgHom_fin_injective (h : n ≤ m) :
    Function.Injective (esymmAlgHom (Fin m) R n) := by
  /-
    R : Type u_3
    n m : Nat
    inst✝ : CommRing R
    h : LE.le n m
    ⊢ Function.Injective ⇑(MvPolynomial.esymmAlgHom (Fin m) R n)
  -/
  rw [injective_iff_map_eq_zero]
  /-
    R : Type u_3
    n m : Nat
    inst✝ : CommRing R
    h : LE.le n m
    ⊢ ∀ (a : MvPolynomial (Fin n) R), Eq ((MvPolynomial.esymmAlgHom (Fin m) R n) a …
  -/
  refine fun p ↦ (fun hp ↦ ?_).mtr
  rw [p.as_sum, map_sum (esymmAlgHom (Fin m) R n), ← Subalgebra.coe_eq_zero,
    AddSubmonoidClass.coe_finset_sum]
  refine sum_ne_zero_of_injOn_supDegree (D := toLex) (support_eq_empty.not.2 hp) (fun t ht ↦ ?_)
    (fun t ht s hs he ↦ DFunLike.ext' <| accumulate_injective h ?_)
  · rw [← esymmAlgHomMonomial, Ne, ← leadingCoeff_eq_zero toLex.injective,
      leadingCoeff_esymmAlgHomMonomial t h]
    /-
      case refine_1
      R : Type u_3
      n m : Nat
      inst✝ : CommRing R
      h : LE.le n m
      p : MvPolynomial (Fin n) R
      hp : Not (Eq p 0)
      t : Finsupp (Fin n) Nat
      ht : Membership.mem p.support t
      ⊢ Not (Eq (MvPolynomial.coeff t p) 0)
    -/
    rwa [mem_support_iff] at ht
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    R : Type u_3
    n m : Nat
    inst✝ : CommRing R
    h : LE.le n m
    p : MvPolynomial (Fin n) R
    hp : Not (Eq p 0)
    t : Finsupp (Fin n) Nat
    ht : Membership.mem (↑p.support) t
    s : Finsupp (Fin n) Nat
    hs : Membership.mem (↑p.support) s
    he : Eq (Function.comp (AddMonoidAlgebra.supDegree ⇑toLex) (fun i => ↑((MvPoly …
    ⊢ Eq ((Fin.accumulate n m) ⇑t) ((Fin.accumulate n m) ⇑s)
  -/
  rw [mem_coe, mem_support_iff] at ht hs
  /-
    case refine_2
    R : Type u_3
    n m : Nat
    inst✝ : CommRing R
    h : LE.le n m
    p : MvPolynomial (Fin n) R
    hp : Not (Eq p 0)
    t : Finsupp (Fin n) Nat
    ht : Ne (MvPolynomial.coeff t p) 0
    s : Finsupp (Fin n) Nat
    hs : Ne (MvPolynomial.coeff s p) 0
    he : Eq (Function.comp (AddMonoidAlgebra.supDegree ⇑toLex) (fun i => ↑((MvPoly …
    ⊢ Eq ((Fin.accumulate n m) ⇑t) ((Fin.accumulate n m) ⇑s)
  -/
  dsimp only [Function.comp] at he
  rwa [← esymmAlgHomMonomial, ← esymmAlgHomMonomial, ← ofLex_inj, DFunLike.ext'_iff,
       supDegree_esymmAlgHomMonomial ht t h, supDegree_esymmAlgHomMonomial hs s h] at he


lemma esymmAlgHom_injective (hn : n ≤ Fintype.card σ) :
    Function.Injective (esymmAlgHom σ R n) := by
  /-
    σ : Type u_1
    R : Type u_3
    n : Nat
    inst✝¹ : Fintype σ
    inst✝ : CommRing R
    hn : LE.le n (Fintype.card σ)
    ⊢ Function.Injective ⇑(MvPolynomial.esymmAlgHom σ R n)
  -/
  rw [← rename_esymmAlgHom (Fintype.equivFin σ).symm, AlgHom.coe_comp]
  /-
    σ : Type u_1
    R : Type u_3
    n : Nat
    inst✝¹ : Fintype σ
    inst✝ : CommRing R
    hn : LE.le n (Fintype.card σ)
    ⊢ Function.Injective (Function.comp ⇑↑(MvPolynomial.renameSymmetricSubalgebra  …
  -/
  exact (AlgEquiv.injective _).comp (esymmAlgHom_fin_injective R hn)
  /-
    🎉 no goals
  -/


lemma esymmAlgHom_fin_bijective (n : ℕ) :
    Function.Bijective (esymmAlgHom (Fin n) R n) := by
  /-
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    ⊢ Function.Bijective ⇑(MvPolynomial.esymmAlgHom (Fin n) R n)
  -/
  use esymmAlgHom_fin_injective R le_rfl
  /-
    case right
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    ⊢ Function.Surjective ⇑(MvPolynomial.esymmAlgHom (Fin n) R n)
  -/
  rintro ⟨p, hp⟩
  /-
    case right.mk
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    p : MvPolynomial (Fin n) R
    hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
    ⊢ Exists fun a => Eq ((MvPolynomial.esymmAlgHom (Fin n) R n) a) ⟨p, hp⟩
  -/
  rw [← AlgHom.mem_range]
  /-
    case right.mk
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    p : MvPolynomial (Fin n) R
    hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
    ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨p, hp⟩
  -/
  obtain rfl | h0 := eq_or_ne p 0
    /-
      case right.mk.inl
      R : Type u_3
      inst✝ : CommRing R
      n : Nat
      hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) 0
      ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨0, hp⟩
    -/
  · exact Subalgebra.zero_mem _
    /-
      🎉 no goals
    -/
  /-
    case right.mk.inr
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    p : MvPolynomial (Fin n) R
    hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
    h0 : Ne p 0
    ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨p, hp⟩
  -/
  induction' he : p.supDegree toLex using WellFoundedLT.induction with t ih generalizing p; subst he
  /-
    case right.mk.inr.ind
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    p : MvPolynomial (Fin n) R
    hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
    h0 : Ne p 0
    ih : ∀ (y : Lex (Finsupp (Fin n) Nat)), LT.lt y (AddMonoidAlgebra.supDegree (⇑ …
    ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨p, hp⟩
  -/
  let t := Finsupp.equivFunOnFinite.symm (invAccumulate n n <| ↑(ofLex <| p.supDegree toLex))
  have hd :
      (esymmAlgHomMonomial _ t <| p.leadingCoeff toLex).supDegree toLex = p.supDegree toLex := by
    rw [← ofLex_inj, DFunLike.ext'_iff, supDegree_esymmAlgHomMonomial _ _ le_rfl]
    · exact accumulate_invAccumulate le_rfl hp.antitone_supDegree
    · rwa [Ne, leadingCoeff_eq_zero toLex.injective]
  /-
    case right.mk.inr.ind
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    p : MvPolynomial (Fin n) R
    hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
    h0 : Ne p 0
    ih : ∀ (y : Lex (Finsupp (Fin n) Nat)), LT.lt y (AddMonoidAlgebra.supDegree (⇑ …
    t : Finsupp (Fin n) Nat := Finsupp.equivFunOnFinite.symm (Fin.invAccumulate n  …
    hd : Eq (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMonomial …
    ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨p, hp⟩
  -/
  obtain he | hne := eq_or_ne p (esymmAlgHomMonomial _ t <| p.leadingCoeff toLex)
    /-
      case right.mk.inr.ind.inl
      R : Type u_3
      inst✝ : CommRing R
      n : Nat
      p : MvPolynomial (Fin n) R
      hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
      h0 : Ne p 0
      ih : ∀ (y : Lex (Finsupp (Fin n) Nat)), LT.lt y (AddMonoidAlgebra.supDegree (⇑ …
      t : Finsupp (Fin n) Nat := Finsupp.equivFunOnFinite.symm (Fin.invAccumulate n  …
      hd : Eq (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMonomial …
      he : Eq p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (AddMonoidAlgebra.leadin …
      ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨p, hp⟩
    -/
  · convert AlgHom.mem_range_self _ (monomial t <| p.leadingCoeff toLex)
    /-
      🎉 no goals
    -/
  /-
    case right.mk.inr.ind.inr
    R : Type u_3
    inst✝ : CommRing R
    n : Nat
    p : MvPolynomial (Fin n) R
    hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
    h0 : Ne p 0
    ih : ∀ (y : Lex (Finsupp (Fin n) Nat)), LT.lt y (AddMonoidAlgebra.supDegree (⇑ …
    t : Finsupp (Fin n) Nat := Finsupp.equivFunOnFinite.symm (Fin.invAccumulate n  …
    hd : Eq (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMonomial …
    hne : Ne p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (AddMonoidAlgebra.leadi …
    ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨p, hp⟩
  -/
  have := (supDegree_sub_lt_of_leadingCoeff_eq toLex.injective hd.symm ?_).resolve_right hne
    /-
      case right.mk.inr.ind.inr.refine_2
      R : Type u_3
      inst✝ : CommRing R
      n : Nat
      p : MvPolynomial (Fin n) R
      hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
      h0 : Ne p 0
      ih : ∀ (y : Lex (Finsupp (Fin n) Nat)), LT.lt y (AddMonoidAlgebra.supDegree (⇑ …
      t : Finsupp (Fin n) Nat := Finsupp.equivFunOnFinite.symm (Fin.invAccumulate n  …
      hd : Eq (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMonomial …
      hne : Ne p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (AddMonoidAlgebra.leadi …
      this : LT.lt (AddMonoidAlgebra.supDegree (⇑toLex) (HSub.hSub p (MvPolynomial.e …
      ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨p, hp⟩
    -/
  · specialize ih _ this _ (Subalgebra.sub_mem _ hp <| isSymmetric_esymmAlgHomMonomial _ _) _ rfl
      /-
        case right.mk.inr.ind.inr.refine_2
        R : Type u_3
        inst✝ : CommRing R
        n : Nat
        p : MvPolynomial (Fin n) R
        hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
        h0 : Ne p 0
        ih : ∀ (y : Lex (Finsupp (Fin n) Nat)), LT.lt y (AddMonoidAlgebra.supDegree (⇑ …
        t : Finsupp (Fin n) Nat := Finsupp.equivFunOnFinite.symm (Fin.invAccumulate n  …
        hd : Eq (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMonomial …
        hne : Ne p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (AddMonoidAlgebra.leadi …
        this : LT.lt (AddMonoidAlgebra.supDegree (⇑toLex) (HSub.hSub p (MvPolynomial.e …
        ⊢ Ne (HSub.hSub p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (AddMonoidAlgebr …
      -/
    · rwa [sub_ne_zero]
      /-
        🎉 no goals
      -/
    /-
      case right.mk.inr.ind.inr.refine_2
      R : Type u_3
      inst✝ : CommRing R
      n : Nat
      p : MvPolynomial (Fin n) R
      hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
      h0 : Ne p 0
      t : Finsupp (Fin n) Nat := Finsupp.equivFunOnFinite.symm (Fin.invAccumulate n  …
      hd : Eq (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMonomial …
      hne : Ne p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (AddMonoidAlgebra.leadi …
      this : LT.lt (AddMonoidAlgebra.supDegree (⇑toLex) (HSub.hSub p (MvPolynomial.e …
      ih : Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨HSub.hSub p  …
      ⊢ Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨p, hp⟩
    -/
    convert ← Subalgebra.add_mem _ ih ⟨monomial t (p.leadingCoeff toLex), rfl⟩
    /-
      case h.e'_5.h.e'_3
      R : Type u_3
      inst✝ : CommRing R
      n : Nat
      p : MvPolynomial (Fin n) R
      hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
      h0 : Ne p 0
      t : Finsupp (Fin n) Nat := Finsupp.equivFunOnFinite.symm (Fin.invAccumulate n  …
      hd : Eq (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMonomial …
      hne : Ne p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (AddMonoidAlgebra.leadi …
      this : LT.lt (AddMonoidAlgebra.supDegree (⇑toLex) (HSub.hSub p (MvPolynomial.e …
      ih : Membership.mem (MvPolynomial.esymmAlgHom (Fin n) R n).range ⟨HSub.hSub p  …
      ⊢ Eq (↑(HAdd.hAdd ⟨HSub.hSub p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (Ad …
    -/
    apply sub_add_cancel p
    /-
      🎉 no goals
    -/
    /-
      case right.mk.inr.ind.inr.refine_1
      R : Type u_3
      inst✝ : CommRing R
      n : Nat
      p : MvPolynomial (Fin n) R
      hp : Membership.mem (MvPolynomial.symmetricSubalgebra (Fin n) R) p
      h0 : Ne p 0
      ih : ∀ (y : Lex (Finsupp (Fin n) Nat)), LT.lt y (AddMonoidAlgebra.supDegree (⇑ …
      t : Finsupp (Fin n) Nat := Finsupp.equivFunOnFinite.symm (Fin.invAccumulate n  …
      hd : Eq (AddMonoidAlgebra.supDegree (⇑toLex) (MvPolynomial.esymmAlgHomMonomial …
      hne : Ne p (MvPolynomial.esymmAlgHomMonomial (Fin n) t (AddMonoidAlgebra.leadi …
      ⊢ Eq (AddMonoidAlgebra.leadingCoeff (⇑toLex) p) (AddMonoidAlgebra.leadingCoeff …
    -/
  · rw [leadingCoeff_esymmAlgHomMonomial t le_rfl]
    /-
      🎉 no goals
    -/


lemma esymmAlgHom_fin_surjective (h : m ≤ n) :
    Function.Surjective (esymmAlgHom (Fin m) R n) := by
  /-
    R : Type u_3
    n m : Nat
    inst✝ : CommRing R
    h : LE.le m n
    ⊢ Function.Surjective ⇑(MvPolynomial.esymmAlgHom (Fin m) R n)
  -/
  intro p
  /-
    R : Type u_3
    n m : Nat
    inst✝ : CommRing R
    h : LE.le m n
    p : Subtype fun x => Membership.mem (MvPolynomial.symmetricSubalgebra (Fin m)  …
    ⊢ Exists fun a => Eq ((MvPolynomial.esymmAlgHom (Fin m) R n) a) p
  -/
  obtain ⟨q, rfl⟩ := (esymmAlgHom_fin_bijective R m).2 p
  /-
    case intro
    R : Type u_3
    n m : Nat
    inst✝ : CommRing R
    h : LE.le m n
    q : MvPolynomial (Fin m) R
    ⊢ Exists fun a => Eq ((MvPolynomial.esymmAlgHom (Fin m) R n) a) ((MvPolynomial …
  -/
  rw [← AlgHom.mem_range]
  induction q using MvPolynomial.induction_on with
  | h_C r => rw [← algebraMap_eq, AlgHom.commutes]; apply Subalgebra.algebraMap_mem
  | h_add p q hp hq => rw [map_add]; exact Subalgebra.add_mem _ hp hq
  | h_X p i hp =>
    rw [map_mul]
    apply Subalgebra.mul_mem _ hp
    rw [AlgHom.mem_range]
    refine ⟨X ⟨i, i.2.trans_le h⟩, ?_⟩
    simp_rw [esymmAlgHom, aeval_X]


lemma esymmAlgHom_surjective (hn : Fintype.card σ ≤ n) :
    Function.Surjective (esymmAlgHom σ R n) := by
  /-
    σ : Type u_1
    R : Type u_3
    n : Nat
    inst✝¹ : Fintype σ
    inst✝ : CommRing R
    hn : LE.le (Fintype.card σ) n
    ⊢ Function.Surjective ⇑(MvPolynomial.esymmAlgHom σ R n)
  -/
  rw [← rename_esymmAlgHom (Fintype.equivFin σ).symm, AlgHom.coe_comp]
  /-
    σ : Type u_1
    R : Type u_3
    n : Nat
    inst✝¹ : Fintype σ
    inst✝ : CommRing R
    hn : LE.le (Fintype.card σ) n
    ⊢ Function.Surjective (Function.comp ⇑↑(MvPolynomial.renameSymmetricSubalgebra …
  -/
  exact (AlgEquiv.surjective _).comp (esymmAlgHom_fin_surjective R hn)
  /-
    🎉 no goals
  -/


/-- If the cardinality of `σ` is `n`, then `esymmAlgHom σ R n` is an isomorphism. -/
@[simps! apply]
noncomputable def esymmAlgEquiv (hn : Fintype.card σ = n) :
    MvPolynomial (Fin n) R ≃ₐ[R] symmetricSubalgebra σ R :=
  AlgEquiv.ofBijective (esymmAlgHom σ R n)
    ⟨esymmAlgHom_injective R hn.ge, esymmAlgHom_surjective R hn.le⟩


