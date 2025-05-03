local notation a " /ℚ " q => (q : ℚ≥0)⁻¹ • a


lemma expect_eq_zero_iff_of_nonneg (hs : s.Nonempty) (hf : ∀ i ∈ s, 0 ≤ f i) :
    𝔼 i ∈ s, f i = 0 ↔ ∀ i ∈ s, f i = 0 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : OrderedAddCommMonoid α
    inst✝ : Module NNRat α
    s : Finset ι
    f : ι → α
    hs : s.Nonempty
    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
    ⊢ Iff (Eq (s.expect fun i => f i) 0) (∀ (i : ι), Membership.mem s i → Eq (f i) …
  -/
  simp [expect, sum_eq_zero_iff_of_nonneg hf, hs.ne_empty]
  /-
    🎉 no goals
  -/


lemma expect_eq_zero_iff_of_nonpos (hs : s.Nonempty) (hf : ∀ i ∈ s, f i ≤ 0) :
    𝔼 i ∈ s, f i = 0 ↔ ∀ i ∈ s, f i = 0 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝¹ : OrderedAddCommMonoid α
    inst✝ : Module NNRat α
    s : Finset ι
    f : ι → α
    hs : s.Nonempty
    hf : ∀ (i : ι), Membership.mem s i → LE.le (f i) 0
    ⊢ Iff (Eq (s.expect fun i => f i) 0) (∀ (i : ι), Membership.mem s i → Eq (f i) …
  -/
  simp [expect, sum_eq_zero_iff_of_nonpos hf, hs.ne_empty]
  /-
    🎉 no goals
  -/


lemma expect_le_expect (hfg : ∀ i ∈ s, f i ≤ g i) : 𝔼 i ∈ s, f i ≤ 𝔼 i ∈ s, g i :=
                                                     /-
                                                       ι : Type u_1
                                                       α : Type u_2
                                                       inst✝² : OrderedAddCommMonoid α
                                                       inst✝¹ : Module NNRat α
                                                       s : Finset ι
                                                       f g : ι → α
                                                       inst✝ : PosSMulMono NNRat α
                                                       hfg : ∀ (i : ι), Membership.mem s i → LE.le (f i) (g i)
                                                       ⊢ LE.le 0 (Inv.inv ↑s.card)
                                                     -/
  smul_le_smul_of_nonneg_left (sum_le_sum hfg) <| by positivity
                                                     /-
                                                       🎉 no goals
                                                     -/


/-- This is a (beta-reduced) version of the standard lemma `Finset.expect_le_expect`,
convenient for the `gcongr` tactic. -/
@[gcongr]
lemma _root_.GCongr.expect_le_expect (h : ∀ i ∈ s, f i ≤ g i) : s.expect f ≤ s.expect g :=
  Finset.expect_le_expect h


lemma expect_le (hs : s.Nonempty) (h : ∀ x ∈ s, f x ≤ a) : 𝔼 i ∈ s, f i ≤ a :=
  (inv_smul_le_iff_of_pos <| mod_cast hs.card_pos).2 <| by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : OrderedAddCommMonoid α
      inst✝¹ : Module NNRat α
      s : Finset ι
      f : ι → α
      inst✝ : PosSMulMono NNRat α
      a : α
      hs : s.Nonempty
      h : ∀ (x : ι), Membership.mem s x → LE.le (f x) a
      ⊢ LE.le (s.sum fun i => (fun i => f i) i) (HSMul.hSMul (↑s.card) a)
    -/
    rw [Nat.cast_smul_eq_nsmul]; exact sum_le_card_nsmul _ _ _ h
                                 /-
                                   🎉 no goals
                                 -/


lemma le_expect (hs : s.Nonempty) (h : ∀ x ∈ s, a ≤ f x) : a ≤ 𝔼 i ∈ s, f i :=
  (le_inv_smul_iff_of_pos <| mod_cast hs.card_pos).2 <| by
    /-
      ι : Type u_1
      α : Type u_2
      inst✝² : OrderedAddCommMonoid α
      inst✝¹ : Module NNRat α
      s : Finset ι
      f : ι → α
      inst✝ : PosSMulMono NNRat α
      a : α
      hs : s.Nonempty
      h : ∀ (x : ι), Membership.mem s x → LE.le a (f x)
      ⊢ LE.le (HSMul.hSMul (↑s.card) a) (s.sum fun i => (fun i => f i) i)
    -/
    rw [Nat.cast_smul_eq_nsmul]; exact card_nsmul_le_sum _ _ _ h
                                 /-
                                   🎉 no goals
                                 -/


lemma expect_nonneg (hf : ∀ i ∈ s, 0 ≤ f i) : 0 ≤ 𝔼 i ∈ s, f i :=
                  /-
                    ι : Type u_1
                    α : Type u_2
                    inst✝² : OrderedAddCommMonoid α
                    inst✝¹ : Module NNRat α
                    s : Finset ι
                    f : ι → α
                    inst✝ : PosSMulMono NNRat α
                    hf : ∀ (i : ι), Membership.mem s i → LE.le 0 (f i)
                    ⊢ LE.le 0 (Inv.inv ↑s.card)
                  -/
  smul_nonneg (by positivity) <| sum_nonneg hf
                  /-
                    🎉 no goals
                  -/


/-- Let `{a | p a}` be an additive subsemigroup of an additive commutative monoid `M`. If `m` is a
subadditive function (`m (a + b) ≤ m a + m b`) preserved under division by a natural, `f` is a
function valued in that subsemigroup and `s` is a nonempty set, then
`m (𝔼 i ∈ s, f i) ≤ 𝔼 i ∈ s, m (f i)`. -/
lemma le_expect_nonempty_of_subadditive_on_pred (h_add : ∀ a b, p a → p b → m (a + b) ≤ m a + m b)
    (hp_add : ∀ a b, p a → p b → p (a + b)) (h_div : ∀ (n : ℕ) a, p a → m (a /ℚ n) = m a /ℚ n)
    (hs_nonempty : s.Nonempty) (hs : ∀ i ∈ s, p (f i)) :
    m (𝔼 i ∈ s, f i) ≤ 𝔼 i ∈ s, m (f i) := by
  /-
    ι : Type u_1
    M : Type u_4
    N : Type u_5
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module NNRat M
    inst✝² : OrderedAddCommMonoid N
    inst✝¹ : Module NNRat N
    inst✝ : PosSMulMono NNRat N
    m : M → N
    p : M → Prop
    f : ι → M
    s : Finset ι
    h_add : ∀ (a b : M), p a → p b → LE.le (m (HAdd.hAdd a b)) (HAdd.hAdd (m a) (m …
    hp_add : ∀ (a b : M), p a → p b → p (HAdd.hAdd a b)
    h_div : ∀ (n : Nat) (a : M), p a → Eq (m (HSMul.hSMul (Inv.inv ↑n) a)) (HSMul. …
    hs_nonempty : s.Nonempty
    hs : ∀ (i : ι), Membership.mem s i → p (f i)
    ⊢ LE.le (m (s.expect fun i => f i)) (s.expect fun i => m (f i))
  -/
  simp only [expect, h_div _ _ (sum_induction_nonempty _ _ hp_add hs_nonempty hs)]
  exact smul_le_smul_of_nonneg_left
    (le_sum_nonempty_of_subadditive_on_pred _ _ h_add hp_add _ _ hs_nonempty hs) <| by positivity


/-- If `m : M → N` is a subadditive function (`m (a + b) ≤ m a + m b`) and `s` is a nonempty set,
then `m (𝔼 i ∈ s, f i) ≤ 𝔼 i ∈ s, m (f i)`. -/
lemma le_expect_nonempty_of_subadditive (m : M → N) (h_mul : ∀ a b, m (a + b) ≤ m a + m b)
    (h_div : ∀ (n : ℕ) a, m (a /ℚ n) = m a /ℚ n) (hs : s.Nonempty) :
    m (𝔼 i ∈ s, f i) ≤ 𝔼 i ∈ s, m (f i) :=
                                                                    /-
                                                                      ι : Type u_1
                                                                      M : Type u_4
                                                                      N : Type u_5
                                                                      inst✝⁴ : AddCommMonoid M
                                                                      inst✝³ : Module NNRat M
                                                                      inst✝² : OrderedAddCommMonoid N
                                                                      inst✝¹ : Module NNRat N
                                                                      inst✝ : PosSMulMono NNRat N
                                                                      f : ι → M
                                                                      s : Finset ι
                                                                      m : M → N
                                                                      h_mul : ∀ (a b : M), LE.le (m (HAdd.hAdd a b)) (HAdd.hAdd (m a) (m b))
                                                                      h_div : ∀ (n : Nat) (a : M), Eq (m (HSMul.hSMul (Inv.inv ↑n) a)) (HSMul.hSMul  …
                                                                      hs : s.Nonempty
                                                                      ⊢ ∀ (a b : M), (fun x => True) a → (fun x => True) b → LE.le (m (HAdd.hAdd a b …
                                                                    -/
                                                                    /-
                                                                      🎉 no goals
                                                                    -/
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/
  le_expect_nonempty_of_subadditive_on_pred (p := fun _ ↦ True) (by simpa) (by simp) (by simpa) hs
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
        /-
          ι : Type u_1
          M : Type u_4
          N : Type u_5
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module NNRat M
          inst✝² : OrderedAddCommMonoid N
          inst✝¹ : Module NNRat N
          inst✝ : PosSMulMono NNRat N
          f : ι → M
          s : Finset ι
          m : M → N
          h_mul : ∀ (a b : M), LE.le (m (HAdd.hAdd a b)) (HAdd.hAdd (m a) (m b))
          h_div : ∀ (n : Nat) (a : M), Eq (m (HSMul.hSMul (Inv.inv ↑n) a)) (HSMul.hSMul  …
          hs : s.Nonempty
          ⊢ ∀ (i : ι), Membership.mem s i → (fun x => True) (f i)
        -/
    (by simp)
        /-
          🎉 no goals
        -/


/-- Let `{a | p a}` be a subsemigroup of a commutative monoid `M`. If `m` is a subadditive function
(`m (x + y) ≤ m x + m y`, `m 0 = 0`) preserved under division by a natural and `f` is a function
valued in that subsemigroup, then `m (𝔼 i ∈ s, f i) ≤ 𝔼 i ∈ s, m (f i)`. -/
lemma le_expect_of_subadditive_on_pred (h_zero : m 0 = 0)
    (h_add : ∀ a b, p a → p b → m (a + b) ≤ m a + m b) (hp_add : ∀ a b, p a → p b → p (a + b))
    (h_div : ∀ (n : ℕ) a, p a → m (a /ℚ n) = m a /ℚ n)
    (hs : ∀ i ∈ s, p (f i)) : m (𝔼 i ∈ s, f i) ≤ 𝔼 i ∈ s, m (f i) := by
  /-
    ι : Type u_1
    M : Type u_4
    N : Type u_5
    inst✝⁴ : AddCommMonoid M
    inst✝³ : Module NNRat M
    inst✝² : OrderedAddCommMonoid N
    inst✝¹ : Module NNRat N
    inst✝ : PosSMulMono NNRat N
    m : M → N
    p : M → Prop
    f : ι → M
    s : Finset ι
    h_zero : Eq (m 0) 0
    h_add : ∀ (a b : M), p a → p b → LE.le (m (HAdd.hAdd a b)) (HAdd.hAdd (m a) (m …
    hp_add : ∀ (a b : M), p a → p b → p (HAdd.hAdd a b)
    h_div : ∀ (n : Nat) (a : M), p a → Eq (m (HSMul.hSMul (Inv.inv ↑n) a)) (HSMul. …
    hs : ∀ (i : ι), Membership.mem s i → p (f i)
    ⊢ LE.le (m (s.expect fun i => f i)) (s.expect fun i => m (f i))
  -/
  obtain rfl | hs_nonempty := s.eq_empty_or_nonempty
    /-
      case inl
      ι : Type u_1
      M : Type u_4
      N : Type u_5
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module NNRat M
      inst✝² : OrderedAddCommMonoid N
      inst✝¹ : Module NNRat N
      inst✝ : PosSMulMono NNRat N
      m : M → N
      p : M → Prop
      f : ι → M
      h_zero : Eq (m 0) 0
      h_add : ∀ (a b : M), p a → p b → LE.le (m (HAdd.hAdd a b)) (HAdd.hAdd (m a) (m …
      hp_add : ∀ (a b : M), p a → p b → p (HAdd.hAdd a b)
      h_div : ∀ (n : Nat) (a : M), p a → Eq (m (HSMul.hSMul (Inv.inv ↑n) a)) (HSMul. …
      hs : ∀ (i : ι), Membership.mem EmptyCollection.emptyCollection i → p (f i)
      ⊢ LE.le (m (EmptyCollection.emptyCollection.expect fun i => f i)) (EmptyCollec …
    -/
  · simp [h_zero]
    /-
      🎉 no goals
    -/
    /-
      case inr
      ι : Type u_1
      M : Type u_4
      N : Type u_5
      inst✝⁴ : AddCommMonoid M
      inst✝³ : Module NNRat M
      inst✝² : OrderedAddCommMonoid N
      inst✝¹ : Module NNRat N
      inst✝ : PosSMulMono NNRat N
      m : M → N
      p : M → Prop
      f : ι → M
      s : Finset ι
      h_zero : Eq (m 0) 0
      h_add : ∀ (a b : M), p a → p b → LE.le (m (HAdd.hAdd a b)) (HAdd.hAdd (m a) (m …
      hp_add : ∀ (a b : M), p a → p b → p (HAdd.hAdd a b)
      h_div : ∀ (n : Nat) (a : M), p a → Eq (m (HSMul.hSMul (Inv.inv ↑n) a)) (HSMul. …
      hs : ∀ (i : ι), Membership.mem s i → p (f i)
      hs_nonempty : s.Nonempty
      ⊢ LE.le (m (s.expect fun i => f i)) (s.expect fun i => m (f i))
    -/
  · exact le_expect_nonempty_of_subadditive_on_pred h_add hp_add h_div hs_nonempty hs
    /-
      🎉 no goals
    -/

-- TODO: Contribute back better docstring to `le_prod_of_submultiplicative`

/-- If `m` is a subadditive function (`m (x + y) ≤ m x + m y`, `m 0 = 0`) preserved under division
by a natural, then `m (𝔼 i ∈ s, f i) ≤ 𝔼 i ∈ s, m (f i)`. -/
lemma le_expect_of_subadditive (h_zero : m 0 = 0) (h_add : ∀ a b, m (a + b) ≤ m a + m b)
    (h_div : ∀ (n : ℕ) a, m (a /ℚ n) = m a /ℚ n) : m (𝔼 i ∈ s, f i) ≤ 𝔼 i ∈ s, m (f i) :=
                                                                  /-
                                                                    ι : Type u_1
                                                                    M : Type u_4
                                                                    N : Type u_5
                                                                    inst✝⁴ : AddCommMonoid M
                                                                    inst✝³ : Module NNRat M
                                                                    inst✝² : OrderedAddCommMonoid N
                                                                    inst✝¹ : Module NNRat N
                                                                    inst✝ : PosSMulMono NNRat N
                                                                    m : M → N
                                                                    f : ι → M
                                                                    s : Finset ι
                                                                    h_zero : Eq (m 0) 0
                                                                    h_add : ∀ (a b : M), LE.le (m (HAdd.hAdd a b)) (HAdd.hAdd (m a) (m b))
                                                                    h_div : ∀ (n : Nat) (a : M), Eq (m (HSMul.hSMul (Inv.inv ↑n) a)) (HSMul.hSMul  …
                                                                    ⊢ ∀ (a b : M), (fun x => True) a → (fun x => True) b → LE.le (m (HAdd.hAdd a b …
                                                                  -/
                                                                  /-
                                                                    🎉 no goals
                                                                  -/
                                                                             /-
                                                                               🎉 no goals
                                                                             -/
  le_expect_of_subadditive_on_pred (p := fun _ ↦ True) h_zero (by simpa) (by simp) (by simpa)
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/
        /-
          ι : Type u_1
          M : Type u_4
          N : Type u_5
          inst✝⁴ : AddCommMonoid M
          inst✝³ : Module NNRat M
          inst✝² : OrderedAddCommMonoid N
          inst✝¹ : Module NNRat N
          inst✝ : PosSMulMono NNRat N
          m : M → N
          f : ι → M
          s : Finset ι
          h_zero : Eq (m 0) 0
          h_add : ∀ (a b : M), LE.le (m (HAdd.hAdd a b)) (HAdd.hAdd (m a) (m b))
          h_div : ∀ (n : Nat) (a : M), Eq (m (HSMul.hSMul (Inv.inv ↑n) a)) (HSMul.hSMul  …
          ⊢ ∀ (i : ι), Membership.mem s i → (fun x => True) (f i)
        -/
    (by simp)
        /-
          🎉 no goals
        -/


lemma expect_pos (hf : ∀ i ∈ s, 0 < f i) (hs : s.Nonempty) : 0 < 𝔼 i ∈ s, f i :=
  smul_pos (inv_pos.2 <| mod_cast hs.card_pos) <| sum_pos hf hs


lemma exists_lt_of_lt_expect (hs : s.Nonempty) (h : a < 𝔼 i ∈ s, f i) : ∃ x ∈ s, a < f x := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedAddCommMonoid α
    inst✝¹ : Module NNRat α
    inst✝ : PosSMulMono NNRat α
    s : Finset ι
    f : ι → α
    a : α
    hs : s.Nonempty
    h : LT.lt a (s.expect fun i => f i)
    ⊢ Exists fun x => And (Membership.mem s x) (LT.lt a (f x))
  -/
  contrapose! h; exact expect_le hs h
                 /-
                   🎉 no goals
                 -/


lemma exists_lt_of_expect_lt (hs : s.Nonempty) (h : 𝔼 i ∈ s, f i < a) : ∃ x ∈ s, f x < a := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝² : LinearOrderedAddCommMonoid α
    inst✝¹ : Module NNRat α
    inst✝ : PosSMulMono NNRat α
    s : Finset ι
    f : ι → α
    a : α
    hs : s.Nonempty
    h : LT.lt (s.expect fun i => f i) a
    ⊢ Exists fun x => And (Membership.mem s x) (LT.lt (f x) a)
  -/
  contrapose! h; exact le_expect hs h
                 /-
                   🎉 no goals
                 -/


lemma abs_expect_le (s : Finset ι) (f : ι → α) : |𝔼 i ∈ s, f i| ≤ 𝔼 i ∈ s, |f i| :=
  le_expect_of_subadditive abs_zero abs_add (fun _ ↦ abs_nnqsmul _)


/-- **Cauchy-Schwarz inequality** in terms of `Finset.expect`. -/
lemma expect_mul_sq_le_sq_mul_sq (s : Finset ι) (f g : ι → R) :
    (𝔼 i ∈ s, f i * g i) ^ 2 ≤ (𝔼 i ∈ s, f i ^ 2) * 𝔼 i ∈ s, g i ^ 2 := by
  /-
    ι : Type u_1
    R : Type u_3
    inst✝³ : LinearOrderedCommSemiring R
    inst✝² : ExistsAddOfLE R
    inst✝¹ : Module NNRat R
    inst✝ : PosSMulMono NNRat R
    s : Finset ι
    f g : ι → R
    ⊢ LE.le (HPow.hPow (s.expect fun i => HMul.hMul (f i) (g i)) 2) (HMul.hMul (s. …
  -/
  simp only [expect, smul_pow, inv_pow, smul_mul_smul_comm, ← sq]
  /-
    ι : Type u_1
    R : Type u_3
    inst✝³ : LinearOrderedCommSemiring R
    inst✝² : ExistsAddOfLE R
    inst✝¹ : Module NNRat R
    inst✝ : PosSMulMono NNRat R
    s : Finset ι
    f g : ι → R
    ⊢ LE.le (HSMul.hSMul (Inv.inv (HPow.hPow (↑s.card) 2)) (HPow.hPow (s.sum fun x …
  -/
  gcongr
  /-
    case hb
    ι : Type u_1
    R : Type u_3
    inst✝³ : LinearOrderedCommSemiring R
    inst✝² : ExistsAddOfLE R
    inst✝¹ : Module NNRat R
    inst✝ : PosSMulMono NNRat R
    s : Finset ι
    f g : ι → R
    ⊢ LE.le (HPow.hPow (s.sum fun x => HMul.hMul (f x) (g x)) 2) (HMul.hMul (s.sum …
  -/
  exact sum_mul_sq_le_sq_mul_sq ..
  /-
    🎉 no goals
  -/


lemma expect_eq_zero_iff_of_nonneg [Nonempty ι] (hf : 0 ≤ f) : 𝔼 i, f i = 0 ↔ f = 0 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝³ : Fintype ι
    inst✝² : OrderedAddCommMonoid α
    inst✝¹ : Module NNRat α
    f : ι → α
    inst✝ : Nonempty ι
    hf : LE.le 0 f
    ⊢ Iff (Eq (Finset.univ.expect fun i => f i) 0) (Eq f 0)
  -/
  simp [expect, sum_eq_zero_iff_of_nonneg hf, univ_nonempty.ne_empty]
  /-
    🎉 no goals
  -/


lemma expect_eq_zero_iff_of_nonpos [Nonempty ι] (hf : f ≤ 0) : 𝔼 i, f i = 0 ↔ f = 0 := by
  /-
    ι : Type u_1
    α : Type u_2
    inst✝³ : Fintype ι
    inst✝² : OrderedAddCommMonoid α
    inst✝¹ : Module NNRat α
    f : ι → α
    inst✝ : Nonempty ι
    hf : LE.le f 0
    ⊢ Iff (Eq (Finset.univ.expect fun i => f i) 0) (Eq f 0)
  -/
  simp [expect, sum_eq_zero_iff_of_nonpos hf, univ_nonempty.ne_empty]
  /-
    🎉 no goals
  -/


attribute [local instance] monadLiftOptionMetaM in
/-- Positivity extension for `Finset.expect`. -/
@[positivity Finset.expect _ _]
def evalFinsetExpect : PositivityExt where eval {u α} zα pα e := do
  match e with
  | ~q(@Finset.expect $ι _ $instα $instmod $s $f) =>
    let i : Q($ι) ← mkFreshExprMVarQ q($ι) .syntheticOpaque
    have body : Q($α) := .betaRev f #[i]
    let rbody ← core zα pα body
    let p_pos : Option Q(0 < $e) := ← (do
      let .positive pbody := rbody | pure none -- Fail if the body is not provably positive
      let .some ps ← proveFinsetNonempty s | pure none
      let .some pα' ← trySynthInstanceQ q(OrderedCancelAddCommMonoid $α) | pure none
      let .some instαordsmul ← trySynthInstanceQ q(PosSMulStrictMono ℚ≥0 $α) | pure none
      assumeInstancesCommute
      let pr : Q(∀ i, 0 < $f i) ← mkLambdaFVars #[i] pbody
      return some q(@expect_pos $ι $α $pα' $instmod $s $f $instαordsmul (fun i _ ↦ $pr i) $ps))
    -- Try to show that the sum is positive
    if let some p_pos := p_pos then
      return .positive p_pos
    -- Fall back to showing that the sum is nonnegative
    else
      let pbody ← rbody.toNonneg
      let pr : Q(∀ i, 0 ≤ $f i) ← mkLambdaFVars #[i] pbody
      let instαordmon ← synthInstanceQ q(OrderedAddCommMonoid $α)
      let instαordsmul ← synthInstanceQ q(PosSMulMono ℚ≥0 $α)
      assumeInstancesCommute
      return .nonnegative q(@expect_nonneg $ι $α $instαordmon $instmod $s $f $instαordsmul
        fun i _ ↦ $pr i)
  | _ => throwError "not Finset.expect"


