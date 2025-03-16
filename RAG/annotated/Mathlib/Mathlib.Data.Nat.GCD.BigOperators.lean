theorem coprime_list_prod_left_iff {l : List ℕ} {k : ℕ} :
    Coprime l.prod k ↔ ∀ n ∈ l, Coprime n k := by
  /-
    l : List Nat
    k : Nat
    ⊢ Iff (l.prod.Coprime k) (∀ (n : Nat), Membership.mem l n → n.Coprime k)
  -/
                  /-
                    🎉 no goals
                  -/
  induction l <;> simp [Nat.coprime_mul_iff_left, *]
                  /-
                    🎉 no goals
                  -/


theorem coprime_list_prod_right_iff {k : ℕ} {l : List ℕ} :
    Coprime k l.prod ↔ ∀ n ∈ l, Coprime k n := by
  /-
    k : Nat
    l : List Nat
    ⊢ Iff (k.Coprime l.prod) (∀ (n : Nat), Membership.mem l n → k.Coprime n)
  -/
  simp_rw [coprime_comm (n := k), coprime_list_prod_left_iff]
  /-
    🎉 no goals
  -/


theorem coprime_multiset_prod_left_iff {m : Multiset ℕ} {k : ℕ} :
    Coprime m.prod k ↔ ∀ n ∈ m, Coprime n k := by
  /-
    m : Multiset Nat
    k : Nat
    ⊢ Iff (m.prod.Coprime k) (∀ (n : Nat), Membership.mem m n → n.Coprime k)
  -/
  induction m using Quotient.inductionOn; simpa using coprime_list_prod_left_iff
                                          /-
                                            🎉 no goals
                                          -/


theorem coprime_multiset_prod_right_iff {k : ℕ} {m : Multiset ℕ} :
    Coprime k m.prod ↔ ∀ n ∈ m, Coprime k n := by
  /-
    k : Nat
    m : Multiset Nat
    ⊢ Iff (k.Coprime m.prod) (∀ (n : Nat), Membership.mem m n → k.Coprime n)
  -/
  induction m using Quotient.inductionOn; simpa using coprime_list_prod_right_iff
                                          /-
                                            🎉 no goals
                                          -/


theorem coprime_prod_left_iff {t : Finset ι} {s : ι → ℕ} {x : ℕ} :
    Coprime (∏ i ∈ t, s i) x ↔ ∀ i ∈ t, Coprime (s i) x := by
  /-
    ι : Type u_1
    t : Finset ι
    s : ι → Nat
    x : Nat
    ⊢ Iff ((t.prod fun i => s i).Coprime x) (∀ (i : ι), Membership.mem t i → (s i) …
  -/
  simpa using coprime_multiset_prod_left_iff (m := t.val.map s)
  /-
    🎉 no goals
  -/


theorem coprime_prod_right_iff {x : ℕ} {t : Finset ι} {s : ι → ℕ} :
    Coprime x (∏ i ∈ t, s i) ↔ ∀ i ∈ t, Coprime x (s i) := by
  /-
    ι : Type u_1
    x : Nat
    t : Finset ι
    s : ι → Nat
    ⊢ Iff (x.Coprime (t.prod fun i => s i)) (∀ (i : ι), Membership.mem t i → x.Cop …
  -/
  simpa using coprime_multiset_prod_right_iff (m := t.val.map s)
  /-
    🎉 no goals
  -/


/-- See `IsCoprime.prod_left` for the corresponding lemma about `IsCoprime` -/
alias ⟨_, Coprime.prod_left⟩ := coprime_prod_left_iff


/-- See `IsCoprime.prod_right` for the corresponding lemma about `IsCoprime` -/
alias ⟨_, Coprime.prod_right⟩ := coprime_prod_right_iff


theorem coprime_fintype_prod_left_iff [Fintype ι] {s : ι → ℕ} {x : ℕ} :
    Coprime (∏ i, s i) x ↔ ∀ i, Coprime (s i) x := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    s : ι → Nat
    x : Nat
    ⊢ Iff ((Finset.univ.prod fun i => s i).Coprime x) (∀ (i : ι), (s i).Coprime x)
  -/
  simp [coprime_prod_left_iff]
  /-
    🎉 no goals
  -/


theorem coprime_fintype_prod_right_iff [Fintype ι] {x : ℕ} {s : ι → ℕ} :
    Coprime x (∏ i, s i) ↔ ∀ i, Coprime x (s i) := by
  /-
    ι : Type u_1
    inst✝ : Fintype ι
    x : Nat
    s : ι → Nat
    ⊢ Iff (x.Coprime (Finset.univ.prod fun i => s i)) (∀ (i : ι), x.Coprime (s i))
  -/
  simp [coprime_prod_right_iff]
  /-
    🎉 no goals
  -/


