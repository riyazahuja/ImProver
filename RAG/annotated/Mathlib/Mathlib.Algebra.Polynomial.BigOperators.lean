set_option backward.isDefEq.lazyProjDelta false in -- See https://github.com/leanprover-community/mathlib4/issues/12535
theorem natDegree_list_sum_le (l : List S[X]) : natDegree l.sum ≤ (l.map natDegree).foldr max 0 :=
                                      /-
                                        S : Type u_1
                                        inst✝ : Semiring S
                                        l : List (Polynomial S)
                                        ⊢ LE.le (Polynomial.natDegree 0) 0
                                      -/
  List.sum_le_foldr_max natDegree (by simp) natDegree_add_le _
                                      /-
                                        🎉 no goals
                                      -/


theorem natDegree_multiset_sum_le (l : Multiset S[X]) :
    natDegree l.sum ≤ (l.map natDegree).foldr max 0 :=
                             /-
                               S : Type u_1
                               inst✝ : Semiring S
                               l : Multiset (Polynomial S)
                               ⊢ ∀ (a : List (Polynomial S)), LE.le (Multiset.sum (Quotient.mk (List.isSetoid …
                             -/
  Quotient.inductionOn l (by simpa using natDegree_list_sum_le)
                             /-
                               🎉 no goals
                             -/


theorem natDegree_sum_le (f : ι → S[X]) :
    natDegree (∑ i ∈ s, f i) ≤ s.fold max 0 (natDegree ∘ f) := by
  /-
    ι : Type w
    s : Finset ι
    S : Type u_1
    inst✝ : Semiring S
    f : ι → Polynomial S
    ⊢ LE.le (s.sum fun i => f i).natDegree (Finset.fold Max.max 0 (Function.comp P …
  -/
  simpa using natDegree_multiset_sum_le (s.val.map f)
  /-
    🎉 no goals
  -/


lemma natDegree_sum_le_of_forall_le {n : ℕ} (f : ι → S[X]) (h : ∀ i ∈ s, natDegree (f i) ≤ n) :
    natDegree (∑ i ∈ s, f i) ≤ n :=
                                                                      /-
                                                                        ι : Type w
                                                                        s : Finset ι
                                                                        S : Type u_1
                                                                        inst✝ : Semiring S
                                                                        n : Nat
                                                                        f : ι → Polynomial S
                                                                        h : ∀ (i : ι), Membership.mem s i → LE.le (f i).natDegree n
                                                                        ⊢ And (LE.le 0 n) (∀ (x : ι), Membership.mem s x → LE.le (Function.comp Polyno …
                                                                      -/
  le_trans (natDegree_sum_le s f) <| (Finset.fold_max_le n).mpr <| by simpa
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem degree_list_sum_le (l : List S[X]) : degree l.sum ≤ (l.map natDegree).maximum := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    l : List (Polynomial S)
    ⊢ LE.le l.sum.degree (List.map Polynomial.natDegree l).maximum
  -/
  by_cases h : l.sum = 0
    /-
      case pos
      S : Type u_1
      inst✝ : Semiring S
      l : List (Polynomial S)
      h : Eq l.sum 0
      ⊢ LE.le l.sum.degree (List.map Polynomial.natDegree l).maximum
    -/
  · simp [h]
    /-
      🎉 no goals
    -/
    /-
      case neg
      S : Type u_1
      inst✝ : Semiring S
      l : List (Polynomial S)
      h : Not (Eq l.sum 0)
      ⊢ LE.le l.sum.degree (List.map Polynomial.natDegree l).maximum
    -/
  · rw [degree_eq_natDegree h]
    suffices (l.map natDegree).maximum = ((l.map natDegree).foldr max 0 : ℕ) by
      rw [this]
      simpa using natDegree_list_sum_le l
    /-
      case neg
      S : Type u_1
      inst✝ : Semiring S
      l : List (Polynomial S)
      h : Not (Eq l.sum 0)
      ⊢ Eq (List.map Polynomial.natDegree l).maximum ↑(List.foldr Max.max 0 (List.ma …
    -/
    rw [← List.foldr_max_of_ne_nil]
      /-
        case neg
        S : Type u_1
        inst✝ : Semiring S
        l : List (Polynomial S)
        h : Not (Eq l.sum 0)
        ⊢ Eq ↑(List.foldr Max.max Bot.bot (List.map Polynomial.natDegree l)) ↑(List.fo …
      -/
    · congr
      /-
        🎉 no goals
      -/
    /-
      case neg
      S : Type u_1
      inst✝ : Semiring S
      l : List (Polynomial S)
      h : Not (Eq l.sum 0)
      ⊢ Ne (List.map Polynomial.natDegree l) List.nil
    -/
    contrapose! h
    /-
      case neg
      S : Type u_1
      inst✝ : Semiring S
      l : List (Polynomial S)
      h : Eq (List.map Polynomial.natDegree l) List.nil
      ⊢ Eq l.sum 0
    -/
    rw [List.map_eq_nil_iff] at h
    /-
      case neg
      S : Type u_1
      inst✝ : Semiring S
      l : List (Polynomial S)
      h : Eq l List.nil
      ⊢ Eq l.sum 0
    -/
    simp [h]
    /-
      🎉 no goals
    -/


theorem natDegree_list_prod_le (l : List S[X]) : natDegree l.prod ≤ (l.map natDegree).sum := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    l : List (Polynomial S)
    ⊢ LE.le l.prod.natDegree (List.map Polynomial.natDegree l).sum
  -/
  induction' l with hd tl IH
    /-
      case nil
      S : Type u_1
      inst✝ : Semiring S
      ⊢ LE.le List.nil.prod.natDegree (List.map Polynomial.natDegree List.nil).sum
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      S : Type u_1
      inst✝ : Semiring S
      hd : Polynomial S
      tl : List (Polynomial S)
      IH : LE.le tl.prod.natDegree (List.map Polynomial.natDegree tl).sum
      ⊢ LE.le (List.cons hd tl).prod.natDegree (List.map Polynomial.natDegree (List. …
    -/
  · simpa using natDegree_mul_le.trans (add_le_add_left IH _)
    /-
      🎉 no goals
    -/


theorem degree_list_prod_le (l : List S[X]) : degree l.prod ≤ (l.map degree).sum := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    l : List (Polynomial S)
    ⊢ LE.le l.prod.degree (List.map Polynomial.degree l).sum
  -/
  induction' l with hd tl IH
    /-
      case nil
      S : Type u_1
      inst✝ : Semiring S
      ⊢ LE.le List.nil.prod.degree (List.map Polynomial.degree List.nil).sum
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      S : Type u_1
      inst✝ : Semiring S
      hd : Polynomial S
      tl : List (Polynomial S)
      IH : LE.le tl.prod.degree (List.map Polynomial.degree tl).sum
      ⊢ LE.le (List.cons hd tl).prod.degree (List.map Polynomial.degree (List.cons h …
    -/
  · simpa using (degree_mul_le _ _).trans (add_le_add_left IH _)
    /-
      🎉 no goals
    -/


theorem coeff_list_prod_of_natDegree_le (l : List S[X]) (n : ℕ) (hl : ∀ p ∈ l, natDegree p ≤ n) :
    coeff (List.prod l) (l.length * n) = (l.map fun p => coeff p n).prod := by
  /-
    S : Type u_1
    inst✝ : Semiring S
    l : List (Polynomial S)
    n : Nat
    hl : ∀ (p : Polynomial S), Membership.mem l p → LE.le p.natDegree n
    ⊢ Eq (l.prod.coeff (HMul.hMul l.length n)) (List.map (fun p => p.coeff n) l).p …
  -/
  induction' l with hd tl IH
    /-
      case nil
      S : Type u_1
      inst✝ : Semiring S
      n : Nat
      hl : ∀ (p : Polynomial S), Membership.mem List.nil p → LE.le p.natDegree n
      ⊢ Eq (List.nil.prod.coeff (HMul.hMul List.nil.length n)) (List.map (fun p => p …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case cons
      S : Type u_1
      inst✝ : Semiring S
      n : Nat
      hd : Polynomial S
      tl : List (Polynomial S)
      IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n) → Eq (t …
      hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
      ⊢ Eq ((List.cons hd tl).prod.coeff (HMul.hMul (List.cons hd tl).length n)) (Li …
    -/
  · have hl' : ∀ p ∈ tl, natDegree p ≤ n := fun p hp => hl p (List.mem_cons_of_mem _ hp)
    /-
      case cons
      S : Type u_1
      inst✝ : Semiring S
      n : Nat
      hd : Polynomial S
      tl : List (Polynomial S)
      IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n) → Eq (t …
      hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
      hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n
      ⊢ Eq ((List.cons hd tl).prod.coeff (HMul.hMul (List.cons hd tl).length n)) (Li …
    -/
    simp only [List.prod_cons, List.map, List.length]
    /-
      case cons
      S : Type u_1
      inst✝ : Semiring S
      n : Nat
      hd : Polynomial S
      tl : List (Polynomial S)
      IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n) → Eq (t …
      hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
      hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n
      ⊢ Eq ((HMul.hMul hd tl.prod).coeff (HMul.hMul (HAdd.hAdd tl.length 1) n)) (HMu …
    -/
    rw [add_mul, one_mul, add_comm, ← IH hl', mul_comm tl.length]
    have h : natDegree tl.prod ≤ n * tl.length := by
      refine (natDegree_list_prod_le _).trans ?_
      rw [← tl.length_map natDegree, mul_comm]
      refine List.sum_le_card_nsmul _ _ ?_
      simpa using hl'
    /-
      case cons
      S : Type u_1
      inst✝ : Semiring S
      n : Nat
      hd : Polynomial S
      tl : List (Polynomial S)
      IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n) → Eq (t …
      hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
      hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n
      h : LE.le tl.prod.natDegree (HMul.hMul n tl.length)
      ⊢ Eq ((HMul.hMul hd tl.prod).coeff (HAdd.hAdd n (HMul.hMul n tl.length))) (HMu …
    -/
    have hdn : natDegree hd ≤ n := hl _ (List.mem_cons_self _ _)
    /-
      case cons
      S : Type u_1
      inst✝ : Semiring S
      n : Nat
      hd : Polynomial S
      tl : List (Polynomial S)
      IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n) → Eq (t …
      hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
      hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n
      h : LE.le tl.prod.natDegree (HMul.hMul n tl.length)
      hdn : LE.le hd.natDegree n
      ⊢ Eq ((HMul.hMul hd tl.prod).coeff (HAdd.hAdd n (HMul.hMul n tl.length))) (HMu …
    -/
    rcases hdn.eq_or_lt with (rfl | hdn')
      /-
        case cons.inl
        S : Type u_1
        inst✝ : Semiring S
        hd : Polynomial S
        tl : List (Polynomial S)
        IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree hd.natDegr …
        hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
        hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree hd.natDegree
        h : LE.le tl.prod.natDegree (HMul.hMul hd.natDegree tl.length)
        hdn : LE.le hd.natDegree hd.natDegree
        ⊢ Eq ((HMul.hMul hd tl.prod).coeff (HAdd.hAdd hd.natDegree (HMul.hMul hd.natDe …
      -/
    · rcases h.eq_or_lt with h' | h'
        /-
          case cons.inl.inl
          S : Type u_1
          inst✝ : Semiring S
          hd : Polynomial S
          tl : List (Polynomial S)
          IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree hd.natDegr …
          hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
          hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree hd.natDegree
          h : LE.le tl.prod.natDegree (HMul.hMul hd.natDegree tl.length)
          hdn : LE.le hd.natDegree hd.natDegree
          h' : Eq tl.prod.natDegree (HMul.hMul hd.natDegree tl.length)
          ⊢ Eq ((HMul.hMul hd tl.prod).coeff (HAdd.hAdd hd.natDegree (HMul.hMul hd.natDe …
        -/
      · rw [← h', coeff_mul_degree_add_degree, leadingCoeff, leadingCoeff]
        /-
          🎉 no goals
        -/
        /-
          case cons.inl.inr
          S : Type u_1
          inst✝ : Semiring S
          hd : Polynomial S
          tl : List (Polynomial S)
          IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree hd.natDegr …
          hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
          hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree hd.natDegree
          h : LE.le tl.prod.natDegree (HMul.hMul hd.natDegree tl.length)
          hdn : LE.le hd.natDegree hd.natDegree
          h' : LT.lt tl.prod.natDegree (HMul.hMul hd.natDegree tl.length)
          ⊢ Eq ((HMul.hMul hd tl.prod).coeff (HAdd.hAdd hd.natDegree (HMul.hMul hd.natDe …
        -/
      · rw [coeff_eq_zero_of_natDegree_lt, coeff_eq_zero_of_natDegree_lt h', mul_zero]
        /-
          case cons.inl.inr
          S : Type u_1
          inst✝ : Semiring S
          hd : Polynomial S
          tl : List (Polynomial S)
          IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree hd.natDegr …
          hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
          hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree hd.natDegree
          h : LE.le tl.prod.natDegree (HMul.hMul hd.natDegree tl.length)
          hdn : LE.le hd.natDegree hd.natDegree
          h' : LT.lt tl.prod.natDegree (HMul.hMul hd.natDegree tl.length)
          ⊢ LT.lt (HMul.hMul hd tl.prod).natDegree (HAdd.hAdd hd.natDegree (HMul.hMul hd …
        -/
        exact natDegree_mul_le.trans_lt (add_lt_add_left h' _)
        /-
          🎉 no goals
        -/
      /-
        case cons.inr
        S : Type u_1
        inst✝ : Semiring S
        n : Nat
        hd : Polynomial S
        tl : List (Polynomial S)
        IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n) → Eq (t …
        hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
        hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n
        h : LE.le tl.prod.natDegree (HMul.hMul n tl.length)
        hdn : LE.le hd.natDegree n
        hdn' : LT.lt hd.natDegree n
        ⊢ Eq ((HMul.hMul hd tl.prod).coeff (HAdd.hAdd n (HMul.hMul n tl.length))) (HMu …
      -/
    · rw [coeff_eq_zero_of_natDegree_lt hdn', coeff_eq_zero_of_natDegree_lt, zero_mul]
      /-
        case cons.inr
        S : Type u_1
        inst✝ : Semiring S
        n : Nat
        hd : Polynomial S
        tl : List (Polynomial S)
        IH : (∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n) → Eq (t …
        hl : ∀ (p : Polynomial S), Membership.mem (List.cons hd tl) p → LE.le p.natDeg …
        hl' : ∀ (p : Polynomial S), Membership.mem tl p → LE.le p.natDegree n
        h : LE.le tl.prod.natDegree (HMul.hMul n tl.length)
        hdn : LE.le hd.natDegree n
        hdn' : LT.lt hd.natDegree n
        ⊢ LT.lt (HMul.hMul hd tl.prod).natDegree (HAdd.hAdd n (HMul.hMul n tl.length))
      -/
      exact natDegree_mul_le.trans_lt (add_lt_add_of_lt_of_le hdn' h)
      /-
        🎉 no goals
      -/


theorem natDegree_multiset_prod_le : t.prod.natDegree ≤ (t.map natDegree).sum :=
                             /-
                               R : Type u
                               inst✝ : CommSemiring R
                               t : Multiset (Polynomial R)
                               ⊢ ∀ (a : List (Polynomial R)), LE.le (Multiset.prod (Quotient.mk (List.isSetoi …
                             -/
  Quotient.inductionOn t (by simpa using natDegree_list_prod_le)
                             /-
                               🎉 no goals
                             -/


theorem natDegree_prod_le : (∏ i ∈ s, f i).natDegree ≤ ∑ i ∈ s, (f i).natDegree := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝ : CommSemiring R
    f : ι → Polynomial R
    ⊢ LE.le (s.prod fun i => f i).natDegree (s.sum fun i => (f i).natDegree)
  -/
  simpa using natDegree_multiset_prod_le (s.1.map f)
  /-
    🎉 no goals
  -/


/-- The degree of a product of polynomials is at most the sum of the degrees,
where the degree of the zero polynomial is ⊥.
-/
theorem degree_multiset_prod_le : t.prod.degree ≤ (t.map Polynomial.degree).sum :=
                             /-
                               R : Type u
                               inst✝ : CommSemiring R
                               t : Multiset (Polynomial R)
                               ⊢ ∀ (a : List (Polynomial R)), LE.le (Multiset.prod (Quotient.mk (List.isSetoi …
                             -/
  Quotient.inductionOn t (by simpa using degree_list_prod_le)
                             /-
                               🎉 no goals
                             -/


theorem degree_prod_le : (∏ i ∈ s, f i).degree ≤ ∑ i ∈ s, (f i).degree := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝ : CommSemiring R
    f : ι → Polynomial R
    ⊢ LE.le (s.prod fun i => f i).degree (s.sum fun i => (f i).degree)
  -/
  simpa only [Multiset.map_map] using degree_multiset_prod_le (s.1.map f)
  /-
    🎉 no goals
  -/


/-- The leading coefficient of a product of polynomials is equal to
the product of the leading coefficients, provided that this product is nonzero.

See `Polynomial.leadingCoeff_multiset_prod` (without the `'`) for a version for integral domains,
where this condition is automatically satisfied.
-/
theorem leadingCoeff_multiset_prod' (h : (t.map leadingCoeff).prod ≠ 0) :
    t.prod.leadingCoeff = (t.map leadingCoeff).prod := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    h : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0
    ⊢ Eq t.prod.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod
  -/
  induction' t using Multiset.induction_on with a t ih; · simp
                                                          /-
                                                            🎉 no goals
                                                          -/
  /-
    case cons
    R : Type u
    inst✝ : CommSemiring R
    t✝ : Multiset (Polynomial R)
    a : Polynomial R
    t : Multiset (Polynomial R)
    ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
    h : Ne (Multiset.map Polynomial.leadingCoeff (Multiset.cons a t)).prod 0
    ⊢ Eq (Multiset.cons a t).prod.leadingCoeff (Multiset.map Polynomial.leadingCoe …
  -/
  simp only [Multiset.map_cons, Multiset.prod_cons] at h ⊢
  /-
    case cons
    R : Type u
    inst✝ : CommSemiring R
    t✝ : Multiset (Polynomial R)
    a : Polynomial R
    t : Multiset (Polynomial R)
    ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
    h : Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod …
    ⊢ Eq (HMul.hMul a t.prod).leadingCoeff (HMul.hMul a.leadingCoeff (Multiset.map …
  -/
  rw [Polynomial.leadingCoeff_mul']
    /-
      case cons
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
      h : Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod …
      ⊢ Eq (HMul.hMul a.leadingCoeff t.prod.leadingCoeff) (HMul.hMul a.leadingCoeff  …
    -/
  · rw [ih]
    /-
      case cons
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
      h : Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod …
      ⊢ Ne (Multiset.map Polynomial.leadingCoeff t).prod 0
    -/
    simp only [ne_eq]
    /-
      case cons
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
      h : Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod …
      ⊢ Not (Eq (Multiset.map Polynomial.leadingCoeff t).prod 0)
    -/
    apply right_ne_zero_of_mul h
    /-
      🎉 no goals
    -/
    /-
      case cons
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
      h : Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod …
      ⊢ Ne (HMul.hMul a.leadingCoeff t.prod.leadingCoeff) 0
    -/
  · rw [ih]
      /-
        case cons
        R : Type u
        inst✝ : CommSemiring R
        t✝ : Multiset (Polynomial R)
        a : Polynomial R
        t : Multiset (Polynomial R)
        ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
        h : Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod …
        ⊢ Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod) 0
      -/
    · exact h
      /-
        🎉 no goals
      -/
    /-
      case cons
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
      h : Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod …
      ⊢ Ne (Multiset.map Polynomial.leadingCoeff t).prod 0
    -/
    simp only [ne_eq, not_false_eq_true]
    /-
      case cons
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map Polynomial.leadingCoeff t).prod 0 → Eq t.prod.leadingCoe …
      h : Ne (HMul.hMul a.leadingCoeff (Multiset.map Polynomial.leadingCoeff t).prod …
      ⊢ Not (Eq (Multiset.map Polynomial.leadingCoeff t).prod 0)
    -/
    apply right_ne_zero_of_mul h
    /-
      🎉 no goals
    -/


/-- The leading coefficient of a product of polynomials is equal to
the product of the leading coefficients, provided that this product is nonzero.

See `Polynomial.leadingCoeff_prod` (without the `'`) for a version for integral domains,
where this condition is automatically satisfied.
-/
theorem leadingCoeff_prod' (h : (∏ i ∈ s, (f i).leadingCoeff) ≠ 0) :
    (∏ i ∈ s, f i).leadingCoeff = ∏ i ∈ s, (f i).leadingCoeff := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝ : CommSemiring R
    f : ι → Polynomial R
    h : Ne (s.prod fun i => (f i).leadingCoeff) 0
    ⊢ Eq (s.prod fun i => f i).leadingCoeff (s.prod fun i => (f i).leadingCoeff)
  -/
  simpa using leadingCoeff_multiset_prod' (s.1.map f) (by simpa using h)
  /-
    🎉 no goals
  -/


/-- The degree of a product of polynomials is equal to
the sum of the degrees, provided that the product of leading coefficients is nonzero.

See `Polynomial.natDegree_multiset_prod` (without the `'`) for a version for integral domains,
where this condition is automatically satisfied.
-/
theorem natDegree_multiset_prod' (h : (t.map fun f => leadingCoeff f).prod ≠ 0) :
    t.prod.natDegree = (t.map fun f => natDegree f).sum := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    h : Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0
    ⊢ Eq t.prod.natDegree (Multiset.map (fun f => f.natDegree) t).sum
  -/
  revert h
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    ⊢ Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0 → Eq t.prod.natDegree ( …
  -/
  refine Multiset.induction_on t ?_ fun a t ih ht => ?_; · simp
                                                           /-
                                                             🎉 no goals
                                                           -/
  /-
    case refine_2
    R : Type u
    inst✝ : CommSemiring R
    t✝ : Multiset (Polynomial R)
    a : Polynomial R
    t : Multiset (Polynomial R)
    ih : Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0 → Eq t.prod.natDegre …
    ht : Ne (Multiset.map (fun f => f.leadingCoeff) (Multiset.cons a t)).prod 0
    ⊢ Eq (Multiset.cons a t).prod.natDegree (Multiset.map (fun f => f.natDegree) ( …
  -/
  rw [Multiset.map_cons, Multiset.prod_cons] at ht ⊢
  /-
    case refine_2
    R : Type u
    inst✝ : CommSemiring R
    t✝ : Multiset (Polynomial R)
    a : Polynomial R
    t : Multiset (Polynomial R)
    ih : Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0 → Eq t.prod.natDegre …
    ht : Ne (HMul.hMul a.leadingCoeff (Multiset.map (fun f => f.leadingCoeff) t).p …
    ⊢ Eq (HMul.hMul a t.prod).natDegree (Multiset.cons a.natDegree (Multiset.map ( …
  -/
  rw [Multiset.sum_cons, Polynomial.natDegree_mul', ih]
    /-
      case refine_2
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0 → Eq t.prod.natDegre …
      ht : Ne (HMul.hMul a.leadingCoeff (Multiset.map (fun f => f.leadingCoeff) t).p …
      ⊢ Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0
    -/
  · apply right_ne_zero_of_mul ht
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0 → Eq t.prod.natDegre …
      ht : Ne (HMul.hMul a.leadingCoeff (Multiset.map (fun f => f.leadingCoeff) t).p …
      ⊢ Ne (HMul.hMul a.leadingCoeff t.prod.leadingCoeff) 0
    -/
  · rwa [Polynomial.leadingCoeff_multiset_prod']
    /-
      case refine_2.h
      R : Type u
      inst✝ : CommSemiring R
      t✝ : Multiset (Polynomial R)
      a : Polynomial R
      t : Multiset (Polynomial R)
      ih : Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0 → Eq t.prod.natDegre …
      ht : Ne (HMul.hMul a.leadingCoeff (Multiset.map (fun f => f.leadingCoeff) t).p …
      ⊢ Ne (Multiset.map Polynomial.leadingCoeff t).prod 0
    -/
    apply right_ne_zero_of_mul ht
    /-
      🎉 no goals
    -/


/-- The degree of a product of polynomials is equal to
the sum of the degrees, provided that the product of leading coefficients is nonzero.

See `Polynomial.natDegree_prod` (without the `'`) for a version for integral domains,
where this condition is automatically satisfied.
-/
theorem natDegree_prod' (h : (∏ i ∈ s, (f i).leadingCoeff) ≠ 0) :
    (∏ i ∈ s, f i).natDegree = ∑ i ∈ s, (f i).natDegree := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝ : CommSemiring R
    f : ι → Polynomial R
    h : Ne (s.prod fun i => (f i).leadingCoeff) 0
    ⊢ Eq (s.prod fun i => f i).natDegree (s.sum fun i => (f i).natDegree)
  -/
  simpa using natDegree_multiset_prod' (s.1.map f) (by simpa using h)
  /-
    🎉 no goals
  -/


theorem natDegree_multiset_prod_of_monic (h : ∀ f ∈ t, Monic f) :
    t.prod.natDegree = (t.map natDegree).sum := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
    ⊢ Eq t.prod.natDegree (Multiset.map Polynomial.natDegree t).sum
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
    a✝ : Nontrivial R
    ⊢ Eq t.prod.natDegree (Multiset.map Polynomial.natDegree t).sum
  -/
  apply natDegree_multiset_prod'
  suffices (t.map fun f => leadingCoeff f).prod = 1 by
    rw [this]
    simp
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
    a✝ : Nontrivial R
    ⊢ Eq (Multiset.map (fun f => f.leadingCoeff) t).prod 1
  -/
  convert prod_replicate (Multiset.card t) (1 : R)
    /-
      case h.e'_2.h.e'_3
      R : Type u
      inst✝ : CommSemiring R
      t : Multiset (Polynomial R)
      h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
      a✝ : Nontrivial R
      ⊢ Eq (Multiset.map (fun f => f.leadingCoeff) t) (Multiset.replicate t.card 1)
    -/
  · simp only [eq_replicate, Multiset.card_map, eq_self_iff_true, true_and]
    /-
      case h.e'_2.h.e'_3
      R : Type u
      inst✝ : CommSemiring R
      t : Multiset (Polynomial R)
      h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
      a✝ : Nontrivial R
      ⊢ ∀ (b : R), Membership.mem (Multiset.map (fun f => f.leadingCoeff) t) b → Eq  …
    -/
    rintro i hi
    /-
      case h.e'_2.h.e'_3
      R : Type u
      inst✝ : CommSemiring R
      t : Multiset (Polynomial R)
      h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
      a✝ : Nontrivial R
      i : R
      hi : Membership.mem (Multiset.map (fun f => f.leadingCoeff) t) i
      ⊢ Eq i 1
    -/
    obtain ⟨i, hi, rfl⟩ := Multiset.mem_map.mp hi
    /-
      case h.e'_2.h.e'_3.intro.intro
      R : Type u
      inst✝ : CommSemiring R
      t : Multiset (Polynomial R)
      h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
      a✝ : Nontrivial R
      i : Polynomial R
      hi✝ : Membership.mem t i
      hi : Membership.mem (Multiset.map (fun f => f.leadingCoeff) t) i.leadingCoeff
      ⊢ Eq i.leadingCoeff 1
    -/
    apply h
    /-
      case h.e'_2.h.e'_3.intro.intro.a
      R : Type u
      inst✝ : CommSemiring R
      t : Multiset (Polynomial R)
      h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
      a✝ : Nontrivial R
      i : Polynomial R
      hi✝ : Membership.mem t i
      hi : Membership.mem (Multiset.map (fun f => f.leadingCoeff) t) i.leadingCoeff
      ⊢ Membership.mem t i
    -/
    assumption
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      R : Type u
      inst✝ : CommSemiring R
      t : Multiset (Polynomial R)
      h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
      a✝ : Nontrivial R
      ⊢ Eq 1 (HPow.hPow 1 t.card)
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem degree_multiset_prod_of_monic [Nontrivial R] (h : ∀ f ∈ t, Monic f) :
    t.prod.degree = (t.map degree).sum := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    t : Multiset (Polynomial R)
    inst✝ : Nontrivial R
    h : ∀ (f : Polynomial R), Membership.mem t f → f.Monic
    ⊢ Eq t.prod.degree (Multiset.map Polynomial.degree t).sum
  -/
  have : t.prod ≠ 0 := Monic.ne_zero <| by simpa using monic_multiset_prod_of_monic _ _ h
  rw [degree_eq_natDegree this, natDegree_multiset_prod_of_monic _ h, Nat.cast_multiset_sum,
    Multiset.map_map, Function.comp_def,
    Multiset.map_congr rfl (fun f hf => (degree_eq_natDegree (h f hf).ne_zero).symm)]


theorem natDegree_prod_of_monic (h : ∀ i ∈ s, (f i).Monic) :
    (∏ i ∈ s, f i).natDegree = ∑ i ∈ s, (f i).natDegree := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝ : CommSemiring R
    f : ι → Polynomial R
    h : ∀ (i : ι), Membership.mem s i → (f i).Monic
    ⊢ Eq (s.prod fun i => f i).natDegree (s.sum fun i => (f i).natDegree)
  -/
  simpa using natDegree_multiset_prod_of_monic (s.1.map f) (by simpa using h)
  /-
    🎉 no goals
  -/


theorem degree_prod_of_monic [Nontrivial R] (h : ∀ i ∈ s, (f i).Monic) :
    (∏ i ∈ s, f i).degree = ∑ i ∈ s, (f i).degree := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝¹ : CommSemiring R
    f : ι → Polynomial R
    inst✝ : Nontrivial R
    h : ∀ (i : ι), Membership.mem s i → (f i).Monic
    ⊢ Eq (s.prod fun i => f i).degree (s.sum fun i => (f i).degree)
  -/
  simpa using degree_multiset_prod_of_monic (s.1.map f) (by simpa using h)
  /-
    🎉 no goals
  -/


theorem coeff_multiset_prod_of_natDegree_le (n : ℕ) (hl : ∀ p ∈ t, natDegree p ≤ n) :
    coeff t.prod ((Multiset.card t) * n) = (t.map fun p => coeff p n).prod := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    n : Nat
    hl : ∀ (p : Polynomial R), Membership.mem t p → LE.le p.natDegree n
    ⊢ Eq (t.prod.coeff (HMul.hMul t.card n)) (Multiset.map (fun p => p.coeff n) t) …
  -/
  induction t using Quotient.inductionOn
  /-
    case h
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    n : Nat
    a✝ : List (Polynomial R)
    hl : ∀ (p : Polynomial R), Membership.mem (Quotient.mk (List.isSetoid (Polynom …
    ⊢ Eq ((Multiset.prod (Quotient.mk (List.isSetoid (Polynomial R)) a✝)).coeff (H …
  -/
  simpa using coeff_list_prod_of_natDegree_le _ _ hl
  /-
    🎉 no goals
  -/


theorem coeff_prod_of_natDegree_le (f : ι → R[X]) (n : ℕ) (h : ∀ p ∈ s, natDegree (f p) ≤ n) :
    coeff (∏ i ∈ s, f i) (#s * n) = ∏ i ∈ s, coeff (f i) n := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝ : CommSemiring R
    f : ι → Polynomial R
    n : Nat
    h : ∀ (p : ι), Membership.mem s p → LE.le (f p).natDegree n
    ⊢ Eq ((s.prod fun i => f i).coeff (HMul.hMul s.card n)) (s.prod fun i => (f i) …
  -/
  cases' s with l hl
  /-
    case mk
    R : Type u
    ι : Type w
    inst✝ : CommSemiring R
    f : ι → Polynomial R
    n : Nat
    l : Multiset ι
    hl : l.Nodup
    h : ∀ (p : ι), Membership.mem { val := l, nodup := hl } p → LE.le (f p).natDeg …
    ⊢ Eq (({ val := l, nodup := hl }.prod fun i => f i).coeff (HMul.hMul { val :=  …
  -/
  convert coeff_multiset_prod_of_natDegree_le (l.map f) n ?_
    /-
      case h.e'_2.h.e'_4.h.e'_5
      R : Type u
      ι : Type w
      inst✝ : CommSemiring R
      f : ι → Polynomial R
      n : Nat
      l : Multiset ι
      hl : l.Nodup
      h : ∀ (p : ι), Membership.mem { val := l, nodup := hl } p → LE.le (f p).natDeg …
      ⊢ Eq { val := l, nodup := hl }.card (Multiset.map f l).card
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      R : Type u
      ι : Type w
      inst✝ : CommSemiring R
      f : ι → Polynomial R
      n : Nat
      l : Multiset ι
      hl : l.Nodup
      h : ∀ (p : ι), Membership.mem { val := l, nodup := hl } p → LE.le (f p).natDeg …
      ⊢ Eq ({ val := l, nodup := hl }.prod fun i => (f i).coeff n) (Multiset.map (fu …
    -/
  · simp
    /-
      🎉 no goals
    -/
    /-
      case mk
      R : Type u
      ι : Type w
      inst✝ : CommSemiring R
      f : ι → Polynomial R
      n : Nat
      l : Multiset ι
      hl : l.Nodup
      h : ∀ (p : ι), Membership.mem { val := l, nodup := hl } p → LE.le (f p).natDeg …
      ⊢ ∀ (p : Polynomial R), Membership.mem (Multiset.map f l) p → LE.le p.natDegre …
    -/
  · simpa using h
    /-
      🎉 no goals
    -/


theorem coeff_zero_multiset_prod : t.prod.coeff 0 = (t.map fun f => coeff f 0).prod := by
  /-
    R : Type u
    inst✝ : CommSemiring R
    t : Multiset (Polynomial R)
    ⊢ Eq (t.prod.coeff 0) (Multiset.map (fun f => f.coeff 0) t).prod
  -/
  refine Multiset.induction_on t ?_ fun a t ht => ?_; · simp
                                                        /-
                                                          🎉 no goals
                                                        -/
  /-
    case refine_2
    R : Type u
    inst✝ : CommSemiring R
    t✝ : Multiset (Polynomial R)
    a : Polynomial R
    t : Multiset (Polynomial R)
    ht : Eq (t.prod.coeff 0) (Multiset.map (fun f => f.coeff 0) t).prod
    ⊢ Eq ((Multiset.cons a t).prod.coeff 0) (Multiset.map (fun f => f.coeff 0) (Mu …
  -/
  rw [Multiset.prod_cons, Multiset.map_cons, Multiset.prod_cons, Polynomial.mul_coeff_zero, ht]
  /-
    🎉 no goals
  -/


theorem coeff_zero_prod : (∏ i ∈ s, f i).coeff 0 = ∏ i ∈ s, (f i).coeff 0 := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝ : CommSemiring R
    f : ι → Polynomial R
    ⊢ Eq ((s.prod fun i => f i).coeff 0) (s.prod fun i => (f i).coeff 0)
  -/
  simpa using coeff_zero_multiset_prod (s.1.map f)
  /-
    🎉 no goals
  -/


theorem multiset_prod_X_sub_C_nextCoeff (t : Multiset R) :
    nextCoeff (t.map fun x => X - C x).prod = -t.sum := by
  /-
    R : Type u
    inst✝ : CommRing R
    t : Multiset R
    ⊢ Eq (Multiset.map (fun x => HSub.hSub Polynomial.X (Polynomial.C x)) t).prod. …
  -/
  rw [nextCoeff_multiset_prod]
    /-
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      ⊢ Eq (Multiset.map (fun i => (HSub.hSub Polynomial.X (Polynomial.C i)).nextCoe …
    -/
  · simp only [nextCoeff_X_sub_C]
    /-
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      ⊢ Eq (Multiset.map (fun x => Neg.neg x) t).sum (Neg.neg t.sum)
    -/
    exact t.sum_hom (-AddMonoidHom.id R)
    /-
      🎉 no goals
    -/
    /-
      case h
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      ⊢ ∀ (i : R), Membership.mem t i → (HSub.hSub Polynomial.X (Polynomial.C i)).Mo …
    -/
  · intros
    /-
      case h
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      i✝ : R
      a✝ : Membership.mem t i✝
      ⊢ (HSub.hSub Polynomial.X (Polynomial.C i✝)).Monic
    -/
    apply monic_X_sub_C
    /-
      🎉 no goals
    -/


theorem prod_X_sub_C_nextCoeff {s : Finset ι} (f : ι → R) :
    nextCoeff (∏ i ∈ s, (X - C (f i))) = -∑ i ∈ s, f i := by
  /-
    R : Type u
    ι : Type w
    inst✝ : CommRing R
    s : Finset ι
    f : ι → R
    ⊢ Eq (s.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (f i))).nextCoeff ( …
  -/
  simpa using multiset_prod_X_sub_C_nextCoeff (s.1.map f)
  /-
    🎉 no goals
  -/


theorem multiset_prod_X_sub_C_coeff_card_pred (t : Multiset R) (ht : 0 < Multiset.card t) :
    (t.map fun x => X - C x).prod.coeff ((Multiset.card t) - 1) = -t.sum := by
  /-
    R : Type u
    inst✝ : CommRing R
    t : Multiset R
    ht : LT.lt 0 t.card
    ⊢ Eq ((Multiset.map (fun x => HSub.hSub Polynomial.X (Polynomial.C x)) t).prod …
  -/
  nontriviality R
  /-
    R : Type u
    inst✝ : CommRing R
    t : Multiset R
    ht : LT.lt 0 t.card
    a✝ : Nontrivial R
    ⊢ Eq ((Multiset.map (fun x => HSub.hSub Polynomial.X (Polynomial.C x)) t).prod …
  -/
  convert multiset_prod_X_sub_C_nextCoeff (by assumption)
  /-
    case h.e'_2
    R : Type u
    inst✝ : CommRing R
    t : Multiset R
    ht : LT.lt 0 t.card
    a✝ : Nontrivial R
    ⊢ Eq ((Multiset.map (fun x => HSub.hSub Polynomial.X (Polynomial.C x)) t).prod …
  -/
  rw [nextCoeff, if_neg]
  /-
    case h.e'_2
    R : Type u
    inst✝ : CommRing R
    t : Multiset R
    ht : LT.lt 0 t.card
    a✝ : Nontrivial R
    ⊢ Eq ((Multiset.map (fun x => HSub.hSub Polynomial.X (Polynomial.C x)) t).prod …
  -/
  swap
    /-
      case h.e'_2.hnc
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      ht : LT.lt 0 t.card
      a✝ : Nontrivial R
      ⊢ Not (Eq (Multiset.map (fun x => HSub.hSub Polynomial.X (Polynomial.C x)) t). …
    -/
  · rw [natDegree_multiset_prod_of_monic]
    /-
      case h.e'_2.hnc
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      ht : LT.lt 0 t.card
      a✝ : Nontrivial R
      ⊢ Not (Eq (Multiset.map Polynomial.natDegree (Multiset.map (fun x => HSub.hSub …
    -/
    swap
      /-
        case h.e'_2.hnc.h
        R : Type u
        inst✝ : CommRing R
        t : Multiset R
        ht : LT.lt 0 t.card
        a✝ : Nontrivial R
        ⊢ ∀ (f : Polynomial R), Membership.mem (Multiset.map (fun x => HSub.hSub Polyn …
      -/
    · simp only [Multiset.mem_map]
      /-
        case h.e'_2.hnc.h
        R : Type u
        inst✝ : CommRing R
        t : Multiset R
        ht : LT.lt 0 t.card
        a✝ : Nontrivial R
        ⊢ ∀ (f : Polynomial R), (Exists fun a => And (Membership.mem t a) (Eq (HSub.hS …
      -/
      rintro _ ⟨_, _, rfl⟩
      /-
        case h.e'_2.hnc.h.intro.intro
        R : Type u
        inst✝ : CommRing R
        t : Multiset R
        ht : LT.lt 0 t.card
        a✝ : Nontrivial R
        w✝ : R
        left✝ : Membership.mem t w✝
        ⊢ (HSub.hSub Polynomial.X (Polynomial.C w✝)).Monic
      -/
      apply monic_X_sub_C
      /-
        🎉 no goals
      -/
    /-
      case h.e'_2.hnc
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      ht : LT.lt 0 t.card
      a✝ : Nontrivial R
      ⊢ Not (Eq (Multiset.map Polynomial.natDegree (Multiset.map (fun x => HSub.hSub …
    -/
    simp_rw [Multiset.sum_eq_zero_iff, Multiset.mem_map]
    /-
      case h.e'_2.hnc
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      ht : LT.lt 0 t.card
      a✝ : Nontrivial R
      ⊢ Not (∀ (x : Nat), (Exists fun a => And (Exists fun a_1 => And (Membership.me …
    -/
    obtain ⟨x, hx⟩ := card_pos_iff_exists_mem.mp ht
    /-
      case h.e'_2.hnc.intro
      R : Type u
      inst✝ : CommRing R
      t : Multiset R
      ht : LT.lt 0 t.card
      a✝ : Nontrivial R
      x : R
      hx : Membership.mem t x
      ⊢ Not (∀ (x : Nat), (Exists fun a => And (Exists fun a_1 => And (Membership.me …
    -/
    exact fun h => one_ne_zero <| h 1 ⟨_, ⟨x, hx, rfl⟩, natDegree_X_sub_C _⟩
    /-
      🎉 no goals
    -/
  /-
    case h.e'_2
    R : Type u
    inst✝ : CommRing R
    t : Multiset R
    ht : LT.lt 0 t.card
    a✝ : Nontrivial R
    ⊢ Eq ((Multiset.map (fun x => HSub.hSub Polynomial.X (Polynomial.C x)) t).prod …
  -/
                                                     /-
                                                       🎉 no goals
                                                     -/
  congr; rw [natDegree_multiset_prod_of_monic] <;> · simp [natDegree_X_sub_C, monic_X_sub_C]
                                                     /-
                                                       🎉 no goals
                                                     -/


theorem prod_X_sub_C_coeff_card_pred (s : Finset ι) (f : ι → R) (hs : 0 < #s) :
    (∏ i ∈ s, (X - C (f i))).coeff (#s - 1) = -∑ i ∈ s, f i := by
  /-
    R : Type u
    ι : Type w
    inst✝ : CommRing R
    s : Finset ι
    f : ι → R
    hs : LT.lt 0 s.card
    ⊢ Eq ((s.prod fun i => HSub.hSub Polynomial.X (Polynomial.C (f i))).coeff (HSu …
  -/
  simpa using multiset_prod_X_sub_C_coeff_card_pred (s.1.map f) (by simpa using hs)
  /-
    🎉 no goals
  -/


@[simp]
lemma natDegree_multiset_prod_X_sub_C_eq_card (s : Multiset R) :
    (s.map (X - C ·)).prod.natDegree = Multiset.card s := by
  /-
    R : Type u
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    s : Multiset R
    ⊢ Eq (Multiset.map (fun x => HSub.hSub Polynomial.X (Polynomial.C x)) s).prod. …
  -/
  rw [natDegree_multiset_prod_of_monic, Multiset.map_map]
  · simp only [(· ∘ ·), natDegree_X_sub_C, Multiset.map_const', Multiset.sum_replicate, smul_eq_mul,
      mul_one]
    /-
      case h
      R : Type u
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      s : Multiset R
      ⊢ ∀ (f : Polynomial R), Membership.mem (Multiset.map (fun x => HSub.hSub Polyn …
    -/
  · exact Multiset.forall_mem_map_iff.2 fun a _ => monic_X_sub_C a
    /-
      🎉 no goals
    -/


/-- The degree of a product of polynomials is equal to
the sum of the degrees, where the degree of the zero polynomial is ⊥.
`[Nontrivial R]` is needed, otherwise for `l = []` we have `⊥` in the LHS and `0` in the RHS.
-/
theorem degree_list_prod [Nontrivial R] (l : List R[X]) : l.prod.degree = (l.map degree).sum :=
  map_list_prod (@degreeMonoidHom R _ _ _) l


/-- The degree of a product of polynomials is equal to
the sum of the degrees.

See `Polynomial.natDegree_prod'` (with a `'`) for a version for commutative semirings,
where additionally, the product of the leading coefficients must be nonzero.
-/
theorem natDegree_prod (h : ∀ i ∈ s, f i ≠ 0) :
    (∏ i ∈ s, f i).natDegree = ∑ i ∈ s, (f i).natDegree := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    f : ι → Polynomial R
    h : ∀ (i : ι), Membership.mem s i → Ne (f i) 0
    ⊢ Eq (s.prod fun i => f i).natDegree (s.sum fun i => (f i).natDegree)
  -/
  nontriviality R
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    f : ι → Polynomial R
    h : ∀ (i : ι), Membership.mem s i → Ne (f i) 0
    a✝ : Nontrivial R
    ⊢ Eq (s.prod fun i => f i).natDegree (s.sum fun i => (f i).natDegree)
  -/
  apply natDegree_prod'
  /-
    case h
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    f : ι → Polynomial R
    h : ∀ (i : ι), Membership.mem s i → Ne (f i) 0
    a✝ : Nontrivial R
    ⊢ Ne (s.prod fun i => (f i).leadingCoeff) 0
  -/
  rw [prod_ne_zero_iff]
  /-
    case h
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    f : ι → Polynomial R
    h : ∀ (i : ι), Membership.mem s i → Ne (f i) 0
    a✝ : Nontrivial R
    ⊢ ∀ (a : ι), Membership.mem s a → Ne (f a).leadingCoeff 0
  -/
  intro x hx; simp [h x hx]
              /-
                🎉 no goals
              -/


theorem natDegree_multiset_prod (h : (0 : R[X]) ∉ t) :
    natDegree t.prod = (t.map natDegree).sum := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    t : Multiset (Polynomial R)
    h : Not (Membership.mem t 0)
    ⊢ Eq t.prod.natDegree (Multiset.map Polynomial.natDegree t).sum
  -/
  nontriviality R
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    t : Multiset (Polynomial R)
    h : Not (Membership.mem t 0)
    a✝ : Nontrivial R
    ⊢ Eq t.prod.natDegree (Multiset.map Polynomial.natDegree t).sum
  -/
  rw [natDegree_multiset_prod']
  /-
    case h
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    t : Multiset (Polynomial R)
    h : Not (Membership.mem t 0)
    a✝ : Nontrivial R
    ⊢ Ne (Multiset.map (fun f => f.leadingCoeff) t).prod 0
  -/
  simp_rw [Ne, Multiset.prod_eq_zero_iff, Multiset.mem_map, leadingCoeff_eq_zero]
  /-
    case h
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    t : Multiset (Polynomial R)
    h : Not (Membership.mem t 0)
    a✝ : Nontrivial R
    ⊢ Not (Exists fun a => And (Membership.mem t a) (Eq a 0))
  -/
  rintro ⟨_, h, rfl⟩
  /-
    case h.intro.intro
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    t : Multiset (Polynomial R)
    h✝ : Not (Membership.mem t 0)
    a✝ : Nontrivial R
    h : Membership.mem t 0
    ⊢ False
  -/
  contradiction
  /-
    🎉 no goals
  -/


/-- The degree of a product of polynomials is equal to
the sum of the degrees, where the degree of the zero polynomial is ⊥.
-/
theorem degree_multiset_prod [Nontrivial R] : t.prod.degree = (t.map fun f => degree f).sum :=
  map_multiset_prod (@degreeMonoidHom R _ _ _) _


/-- The degree of a product of polynomials is equal to
the sum of the degrees, where the degree of the zero polynomial is ⊥.
-/
theorem degree_prod [Nontrivial R] : (∏ i ∈ s, f i).degree = ∑ i ∈ s, (f i).degree :=
  map_prod (@degreeMonoidHom R _ _ _) _ _


/-- The leading coefficient of a product of polynomials is equal to
the product of the leading coefficients.

See `Polynomial.leadingCoeff_multiset_prod'` (with a `'`) for a version for commutative semirings,
where additionally, the product of the leading coefficients must be nonzero.
-/
theorem leadingCoeff_multiset_prod :
    t.prod.leadingCoeff = (t.map fun f => leadingCoeff f).prod := by
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    t : Multiset (Polynomial R)
    ⊢ Eq t.prod.leadingCoeff (Multiset.map (fun f => f.leadingCoeff) t).prod
  -/
  rw [← leadingCoeffHom_apply, MonoidHom.map_multiset_prod]
  /-
    R : Type u
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    t : Multiset (Polynomial R)
    ⊢ Eq (Multiset.map (⇑Polynomial.leadingCoeffHom) t).prod (Multiset.map (fun f  …
  -/
  simp only [leadingCoeffHom_apply]
  /-
    🎉 no goals
  -/


/-- The leading coefficient of a product of polynomials is equal to
the product of the leading coefficients.

See `Polynomial.leadingCoeff_prod'` (with a `'`) for a version for commutative semirings,
where additionally, the product of the leading coefficients must be nonzero.
-/
theorem leadingCoeff_prod : (∏ i ∈ s, f i).leadingCoeff = ∏ i ∈ s, (f i).leadingCoeff := by
  /-
    R : Type u
    ι : Type w
    s : Finset ι
    inst✝¹ : CommSemiring R
    inst✝ : NoZeroDivisors R
    f : ι → Polynomial R
    ⊢ Eq (s.prod fun i => f i).leadingCoeff (s.prod fun i => (f i).leadingCoeff)
  -/
  simpa using leadingCoeff_multiset_prod (s.1.map f)
  /-
    🎉 no goals
  -/


