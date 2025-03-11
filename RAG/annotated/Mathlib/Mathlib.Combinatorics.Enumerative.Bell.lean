/-- Number of partitions of a set of cardinality `m.sum`
whose parts have cardinalities given by `m` -/
def bell (m : Multiset ℕ) : ℕ :=
  Nat.multinomial m.toFinset (fun k ↦ k * m.count k) *
    ∏ k ∈ m.toFinset.erase 0, ∏ j ∈ .range (m.count k), (j * k + k - 1).choose (k - 1)


@[simp]
theorem bell_zero : bell 0 = 1 := rfl


private theorem bell_mul_eq_lemma {x : ℕ} (hx : x ≠ 0) :
    ∀ c, x ! ^ c * c ! * ∏ j ∈ Finset.range c, (j * x + x - 1).choose (x - 1) = (x * c)!
            /-
              x : Nat
              hx : Ne x 0
              ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow x.factorial 0) (Nat.factorial 0)) ((Fins …
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
  | c + 1 => calc
      x ! ^ (c + 1) * (c + 1)! * ∏ j ∈ Finset.range (c + 1), (j * x + x - 1).choose (x - 1)
        = x ! * (c + 1) * x ! ^ c * c ! *
            ∏ j ∈ Finset.range (c + 1), (j * x + x - 1).choose (x - 1) := by
        /-
          x : Nat
          hx : Ne x 0
          c : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul (HPow.hPow x.factorial (HAdd.hAdd c 1)) (HAdd.hAdd  …
        -/
        rw [factorial_succ, pow_succ]; ring
                                       /-
                                         🎉 no goals
                                       -/
      _ = (x ! ^ c * c ! * ∏ j in Finset.range c, (j * x + x - 1).choose (x - 1)) *
            (c * x + x - 1).choose (x - 1) * x ! * (c + 1)  := by
        /-
          x : Nat
          hx : Ne x 0
          c : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul x.factorial (HAdd.hAdd c 1))  …
        -/
        rw [Finset.prod_range_succ]; ring
                                     /-
                                       🎉 no goals
                                     -/
      _ = (c + 1) * (c * x + x - 1).choose (x - 1) * (x * c)! * x ! := by
        /-
          x : Nat
          hx : Ne x 0
          c : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hPow x.facto …
        -/
        rw [bell_mul_eq_lemma hx]; ring
                                   /-
                                     🎉 no goals
                                   -/
      _ = (x * (c + 1))! := by
        /-
          x : Nat
          hx : Ne x 0
          c : Nat
          ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HAdd.hAdd c 1) ((HSub.hSub (HAdd.hAdd ( …
        -/
        rw [← Nat.choose_mul_add hx, mul_comm c x, Nat.add_choose_mul_factorial_mul_factorial]
        /-
          x : Nat
          hx : Ne x 0
          c : Nat
          ⊢ Eq (HAdd.hAdd (HMul.hMul x c) x).factorial (HMul.hMul x (HAdd.hAdd c 1)).fac …
        -/
        ring_nf
        /-
          🎉 no goals
        -/


theorem bell_mul_eq (m : Multiset ℕ) :
    m.bell * (m.map (fun j ↦ j !)).prod * ∏ j ∈ (m.toFinset.erase 0), (m.count j)!
      = m.sum ! := by
  /-
    m : Multiset Nat
    ⊢ Eq (HMul.hMul (HMul.hMul m.bell (Multiset.map (fun j => j.factorial) m).prod …
  -/
  unfold bell
  /-
    m : Multiset Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (Nat.multinomial m.toFinset fun k => HMu …
  -/
  rw [← Nat.mul_right_inj (a := ∏ i ∈ m.toFinset, (i * count i m)!) (by positivity)]
  /-
    m : Multiset Nat
    ⊢ Eq (HMul.hMul (m.toFinset.prod fun i => (HMul.hMul i (Multiset.count i m)).f …
  -/
  simp only [← mul_assoc]
  /-
    m : Multiset Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (HMul.hMul (m.toFinset.prod fun i => (HM …
  -/
  rw [Nat.multinomial_spec]
  /-
    m : Multiset Nat
    ⊢ Eq (HMul.hMul (HMul.hMul (HMul.hMul (m.toFinset.sum fun i => HMul.hMul i (Mu …
  -/
  simp only [mul_assoc]
  /-
    m : Multiset Nat
    ⊢ Eq (HMul.hMul (m.toFinset.sum fun i => HMul.hMul i (Multiset.count i m)).fac …
  -/
  rw [mul_comm]
  /-
    m : Multiset Nat
    ⊢ Eq (HMul.hMul (HMul.hMul ((m.toFinset.erase 0).prod fun k => (Finset.range ( …
  -/
  apply congr_arg₂
    /-
      case hx
      m : Multiset Nat
      ⊢ Eq (HMul.hMul ((m.toFinset.erase 0).prod fun k => (Finset.range (Multiset.co …
    -/
  · rw [mul_comm, mul_assoc, ← Finset.prod_mul_distrib, Finset.prod_multiset_map_count]
    suffices this : _ by
      by_cases hm : 0 ∈ m.toFinset
      · rw [← Finset.prod_erase_mul _ _ hm]
        rw [← Finset.prod_erase_mul _ _ hm]
        simp only [factorial_zero, one_pow, mul_one, zero_mul]
        exact this
      · nth_rewrite 1 [← Finset.erase_eq_of_not_mem hm]
        nth_rewrite 3 [← Finset.erase_eq_of_not_mem hm]
        exact this
    /-
      case hx
      m : Multiset Nat
      ⊢ Eq (HMul.hMul ((m.toFinset.erase 0).prod fun x => HPow.hPow x.factorial (Mul …
    -/
    rw [← Finset.prod_mul_distrib]
    /-
      case hx
      m : Multiset Nat
      ⊢ Eq ((m.toFinset.erase 0).prod fun x => HMul.hMul (HPow.hPow x.factorial (Mul …
    -/
    apply Finset.prod_congr rfl
    /-
      case hx
      m : Multiset Nat
      ⊢ ∀ (x : Nat), Membership.mem (m.toFinset.erase 0) x → Eq (HMul.hMul (HPow.hPo …
    -/
    intro x hx
    /-
      case hx
      m : Multiset Nat
      x : Nat
      hx : Membership.mem (m.toFinset.erase 0) x
      ⊢ Eq (HMul.hMul (HPow.hPow x.factorial (Multiset.count x m)) (HMul.hMul (Multi …
    -/
    rw [← mul_assoc, bell_mul_eq_lemma]
    /-
      case hx.hx
      m : Multiset Nat
      x : Nat
      hx : Membership.mem (m.toFinset.erase 0) x
      ⊢ Ne x 0
    -/
    simp only [Finset.mem_erase, ne_eq, mem_toFinset] at hx
    /-
      case hx.hx
      m : Multiset Nat
      x : Nat
      hx : And (Not (Eq x 0)) (Membership.mem m x)
      ⊢ Ne x 0
    -/
    simp only [ne_eq, hx.1, not_false_eq_true]
    /-
      🎉 no goals
    -/
    /-
      case hy
      m : Multiset Nat
      ⊢ Eq (m.toFinset.sum fun i => HMul.hMul i (Multiset.count i m)).factorial m.su …
    -/
  · apply congr_arg
    /-
      case hy.h
      m : Multiset Nat
      ⊢ Eq (m.toFinset.sum fun i => HMul.hMul i (Multiset.count i m)) m.sum
    -/
    rw [Finset.sum_multiset_count]
    /-
      case hy.h
      m : Multiset Nat
      ⊢ Eq (m.toFinset.sum fun i => HMul.hMul i (Multiset.count i m)) (m.toFinset.su …
    -/
    simp only [smul_eq_mul, mul_comm]
    /-
      🎉 no goals
    -/


theorem bell_eq (m : Multiset ℕ) :
    m.bell = m.sum ! / ((m.map (fun j ↦ j !)).prod *
      ∏ j ∈ (m.toFinset.erase 0), (m.count j)!) := by
  /-
    m : Multiset Nat
    ⊢ Eq m.bell (HDiv.hDiv m.sum.factorial (HMul.hMul (Multiset.map (fun j => j.fa …
  -/
  rw [← Nat.mul_left_inj, Nat.div_mul_cancel _]
    /-
      m : Multiset Nat
      ⊢ Eq (HMul.hMul m.bell (HMul.hMul (Multiset.map (fun j => j.factorial) m).prod …
    -/
  · rw [← mul_assoc]
    /-
      m : Multiset Nat
      ⊢ Eq (HMul.hMul (HMul.hMul m.bell (Multiset.map (fun j => j.factorial) m).prod …
    -/
    exact bell_mul_eq m
    /-
      🎉 no goals
    -/
    /-
      m : Multiset Nat
      ⊢ Dvd.dvd (HMul.hMul (Multiset.map (fun j => j.factorial) m).prod ((m.toFinset …
    -/
  · rw [← bell_mul_eq, mul_assoc]
    /-
      m : Multiset Nat
      ⊢ Dvd.dvd (HMul.hMul (Multiset.map (fun j => j.factorial) m).prod ((m.toFinset …
    -/
    apply Nat.dvd_mul_left
    /-
      🎉 no goals
    -/
    /-
      m : Multiset Nat
      ⊢ Ne (HMul.hMul (Multiset.map (fun j => j.factorial) m).prod ((m.toFinset.eras …
    -/
  · rw [← Nat.pos_iff_ne_zero]
    /-
      m : Multiset Nat
      ⊢ LT.lt 0 (HMul.hMul (Multiset.map (fun j => j.factorial) m).prod ((m.toFinset …
    -/
    apply Nat.mul_pos
    · simp only [gt_iff_lt, CanonicallyOrderedCommSemiring.multiset_prod_pos, mem_map,
      forall_exists_index, and_imp, forall_apply_eq_imp_iff₂]
      /-
        case ha
        m : Multiset Nat
        ⊢ ∀ (a : Nat), Membership.mem m a → LT.lt 0 a.factorial
      -/
      exact fun _ _ ↦ Nat.factorial_pos _
      /-
        🎉 no goals
      -/
      /-
        case hb
        m : Multiset Nat
        ⊢ GT.gt ((m.toFinset.erase 0).prod fun j => (Multiset.count j m).factorial) 0
      -/
    · apply Finset.prod_pos
      /-
        case hb.h0
        m : Multiset Nat
        ⊢ ∀ (i : Nat), Membership.mem (m.toFinset.erase 0) i → LT.lt 0 (Multiset.count …
      -/
      exact fun _ _ ↦ Nat.factorial_pos _
      /-
        🎉 no goals
      -/


/-- Number of possibilities of dividing a set with `m * n` elements into `m` groups
of `n`-element subsets. -/
def uniformBell (m n : ℕ) : ℕ := bell (replicate m n)


theorem uniformBell_eq (m n : ℕ) : m.uniformBell n =
    ∏ p ∈ (Finset.range m), Nat.choose (p * n + n - 1) (n - 1) := by
  /-
    m n : Nat
    ⊢ Eq (m.uniformBell n) ((Finset.range m).prod fun p => (HSub.hSub (HAdd.hAdd ( …
  -/
  unfold uniformBell bell
  /-
    m n : Nat
    ⊢ Eq (HMul.hMul (Nat.multinomial (Multiset.replicate m n).toFinset fun k => HM …
  -/
  rw [toFinset_replicate]
  /-
    m n : Nat
    ⊢ Eq (HMul.hMul (Nat.multinomial (ite (Eq m 0) EmptyCollection.emptyCollection …
  -/
  split_ifs with hm
    /-
      case pos
      m n : Nat
      hm : Eq m 0
      ⊢ Eq (HMul.hMul (Nat.multinomial EmptyCollection.emptyCollection fun k => HMul …
    -/
  · simp  [hm]
    /-
      🎉 no goals
    -/
    /-
      case neg
      m n : Nat
      hm : Not (Eq m 0)
      ⊢ Eq (HMul.hMul (Nat.multinomial (Singleton.singleton n) fun k => HMul.hMul k  …
    -/
  · by_cases hn : n = 0
      /-
        case pos
        m n : Nat
        hm : Not (Eq m 0)
        hn : Eq n 0
        ⊢ Eq (HMul.hMul (Nat.multinomial (Singleton.singleton n) fun k => HMul.hMul k  …
      -/
    · simp [hn]
      /-
        🎉 no goals
      -/
      /-
        case neg
        m n : Nat
        hm : Not (Eq m 0)
        hn : Not (Eq n 0)
        ⊢ Eq (HMul.hMul (Nat.multinomial (Singleton.singleton n) fun k => HMul.hMul k  …
      -/
    · rw [show ({n} : Finset ℕ).erase 0 = {n} by simp [Ne.symm hn]]
      /-
        case neg
        m n : Nat
        hm : Not (Eq m 0)
        hn : Not (Eq n 0)
        ⊢ Eq (HMul.hMul (Nat.multinomial (Singleton.singleton n) fun k => HMul.hMul k  …
      -/
      simp [count_replicate]
      /-
        🎉 no goals
      -/


theorem uniformBell_zero_left (n : ℕ) : uniformBell 0 n = 1 := by
  /-
    n : Nat
    ⊢ Eq (Nat.uniformBell 0 n) 1
  -/
  simp [uniformBell_eq]
  /-
    🎉 no goals
  -/


theorem uniformBell_zero_right (m : ℕ) : uniformBell m 0 = 1 := by
  /-
    m : Nat
    ⊢ Eq (m.uniformBell 0) 1
  -/
  simp [uniformBell_eq]
  /-
    🎉 no goals
  -/


theorem uniformBell_succ_left (m n : ℕ) :
    uniformBell (m+1) n = choose (m * n + n - 1) (n - 1) * uniformBell m n := by
  /-
    m n : Nat
    ⊢ Eq ((HAdd.hAdd m 1).uniformBell n) (HMul.hMul ((HSub.hSub (HAdd.hAdd (HMul.h …
  -/
  simp only [uniformBell_eq, Finset.prod_range_succ, mul_comm]
  /-
    🎉 no goals
  -/


theorem uniformBell_one_left (n : ℕ) : uniformBell 1 n = 1 := by
  simp only [uniformBell_eq, Finset.range_one, Finset.prod_singleton, zero_mul,
    zero_add, choose_self]


theorem uniformBell_one_right (m : ℕ) : uniformBell m 1 = 1 := by
  simp only [uniformBell_eq, mul_one, add_tsub_cancel_right, ge_iff_le, le_refl,
    tsub_eq_zero_of_le, choose_zero_right, Finset.prod_const_one]


theorem uniformBell_mul_eq (m : ℕ) {n : ℕ} (hn : n ≠ 0) :
    uniformBell m n * n ! ^ m * m ! = (m * n)! := by
  /-
    m n : Nat
    hn : Ne n 0
    ⊢ Eq (HMul.hMul (HMul.hMul (m.uniformBell n) (HPow.hPow n.factorial m)) m.fact …
  -/
  convert bell_mul_eq (replicate m n)
    /-
      case h.e'_2.h.e'_5.h.e'_6
      m n : Nat
      hn : Ne n 0
      ⊢ Eq (HPow.hPow n.factorial m) (Multiset.map (fun j => j.factorial) (Multiset. …
    -/
  · simp only [map_replicate, prod_replicate]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_2.h.e'_6
      m n : Nat
      hn : Ne n 0
      ⊢ Eq m.factorial (((Multiset.replicate m n).toFinset.erase 0).prod fun j => (M …
    -/
  · simp only [toFinset_replicate]
    /-
      case h.e'_2.h.e'_6
      m n : Nat
      hn : Ne n 0
      ⊢ Eq m.factorial (((ite (Eq m 0) EmptyCollection.emptyCollection (Singleton.si …
    -/
    split_ifs with hm
      /-
        case pos
        m n : Nat
        hn : Ne n 0
        hm : Eq m 0
        ⊢ Eq m.factorial ((EmptyCollection.emptyCollection.erase 0).prod fun x => (Mul …
      -/
    · rw [hm, factorial_zero, eq_comm]
      /-
        case pos
        m n : Nat
        hn : Ne n 0
        hm : Eq m 0
        ⊢ Eq ((EmptyCollection.emptyCollection.erase 0).prod fun x => (Multiset.count  …
      -/
      rw [show (∅ : Finset ℕ).erase 0 = ∅ from rfl,  Finset.prod_empty]
      /-
        🎉 no goals
      -/
      /-
        case neg
        m n : Nat
        hn : Ne n 0
        hm : Not (Eq m 0)
        ⊢ Eq m.factorial (((Singleton.singleton n).erase 0).prod fun x => (Multiset.co …
      -/
    · rw [show ({n} : Finset ℕ).erase 0 = {n} by simp [Ne.symm hn]]
      /-
        case neg
        m n : Nat
        hn : Ne n 0
        hm : Not (Eq m 0)
        ⊢ Eq m.factorial ((Singleton.singleton n).prod fun x => (Multiset.count x (Mul …
      -/
      simp only [Finset.prod_singleton, count_replicate_self]
      /-
        🎉 no goals
      -/
    /-
      case h.e'_3.h.e'_1
      m n : Nat
      hn : Ne n 0
      ⊢ Eq (HMul.hMul m n) (Multiset.replicate m n).sum
    -/
  · simp
    /-
      🎉 no goals
    -/


theorem uniformBell_eq_div (m : ℕ) {n : ℕ} (hn : n ≠ 0) :
    uniformBell m n = (m * n) ! / (n ! ^ m * m !) := by
  /-
    m n : Nat
    hn : Ne n 0
    ⊢ Eq (m.uniformBell n) (HDiv.hDiv (HMul.hMul m n).factorial (HMul.hMul (HPow.h …
  -/
  rw [eq_comm]
  /-
    m n : Nat
    hn : Ne n 0
    ⊢ Eq (HDiv.hDiv (HMul.hMul m n).factorial (HMul.hMul (HPow.hPow n.factorial m) …
  -/
  apply Nat.div_eq_of_eq_mul_left
    /-
      case H1
      m n : Nat
      hn : Ne n 0
      ⊢ LT.lt 0 (HMul.hMul (HPow.hPow n.factorial m) m.factorial)
    -/
  · exact Nat.mul_pos (Nat.pow_pos (Nat.factorial_pos n)) m.factorial_pos
    /-
      🎉 no goals
    -/
    /-
      case H2
      m n : Nat
      hn : Ne n 0
      ⊢ Eq (HMul.hMul m n).factorial (HMul.hMul (m.uniformBell n) (HMul.hMul (HPow.h …
    -/
  · rw [← mul_assoc, ← uniformBell_mul_eq _ hn]
    /-
      🎉 no goals
    -/


