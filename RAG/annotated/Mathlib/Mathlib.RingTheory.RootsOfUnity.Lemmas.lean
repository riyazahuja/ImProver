/-- If `μ` is a primitive `n`th root of unity in `R`, then `∏(1≤k<n) (1-μ^k) = n`.
(Stated with `n+1` in place of `n` to avoid the condition `n ≠ 0`.) -/
lemma prod_one_sub_pow_eq_order {n : ℕ} {μ : R} (hμ : IsPrimitiveRoot μ (n + 1)) :
    ∏ k ∈ range n, (1 - μ ^ (k + 1)) = n + 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    μ : R
    hμ : IsPrimitiveRoot μ (HAdd.hAdd n 1)
    ⊢ Eq ((Finset.range n).prod fun k => HSub.hSub 1 (HPow.hPow μ (HAdd.hAdd k 1)) …
  -/
  have := X_pow_sub_C_eq_prod hμ n.zero_lt_succ (one_pow (n + 1))
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    μ : R
    hμ : IsPrimitiveRoot μ (HAdd.hAdd n 1)
    this : Eq (HSub.hSub (HPow.hPow Polynomial.X (HAdd.hAdd n 1)) (Polynomial.C 1) …
    ⊢ Eq ((Finset.range n).prod fun k => HSub.hSub 1 (HPow.hPow μ (HAdd.hAdd k 1)) …
  -/
  rw [C_1, ← mul_geom_sum, prod_range_succ', pow_zero, mul_one, mul_comm, eq_comm] at this
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    μ : R
    hμ : IsPrimitiveRoot μ (HAdd.hAdd n 1)
    this : Eq (HMul.hMul ((Finset.range n).prod fun k => HSub.hSub Polynomial.X (P …
    ⊢ Eq ((Finset.range n).prod fun k => HSub.hSub 1 (HPow.hPow μ (HAdd.hAdd k 1)) …
  -/
  replace this := mul_right_cancel₀ (Polynomial.X_sub_C_ne_zero 1) this
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    μ : R
    hμ : IsPrimitiveRoot μ (HAdd.hAdd n 1)
    this : Eq ((Finset.range n).prod fun k => HSub.hSub Polynomial.X (Polynomial.C …
    ⊢ Eq ((Finset.range n).prod fun k => HSub.hSub 1 (HPow.hPow μ (HAdd.hAdd k 1)) …
  -/
  apply_fun Polynomial.eval 1 at this
  simpa only [mul_one, map_pow, eval_prod, eval_sub, eval_X, eval_pow, eval_C, eval_geom_sum,
    one_pow, sum_const, card_range, nsmul_eq_mul, Nat.cast_add, Nat.cast_one] using this


/-- If `μ` is a primitive `n`th root of unity in `R`, then `(-1)^(n-1) * ∏(1≤k<n) (μ^k-1) = n`.
(Stated with `n+1` in place of `n` to avoid the condition `n ≠ 0`.) -/
lemma prod_pow_sub_one_eq_order {n : ℕ} {μ : R} (hμ : IsPrimitiveRoot μ (n + 1)) :
    (-1) ^ n * ∏ k ∈ range n, (μ ^ (k + 1) - 1) = n + 1 := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    μ : R
    hμ : IsPrimitiveRoot μ (HAdd.hAdd n 1)
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) n) ((Finset.range n).prod fun k => HSub.hSub ( …
  -/
  have : (-1 : R) ^ n = ∏ k ∈ range n, -1 := by rw [prod_const, card_range]
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    n : Nat
    μ : R
    hμ : IsPrimitiveRoot μ (HAdd.hAdd n 1)
    this : Eq (HPow.hPow (-1) n) ((Finset.range n).prod fun k => -1)
    ⊢ Eq (HMul.hMul (HPow.hPow (-1) n) ((Finset.range n).prod fun k => HSub.hSub ( …
  -/
  simp only [this, ← prod_mul_distrib, neg_one_mul, neg_sub, ← prod_one_sub_pow_eq_order hμ]
  /-
    🎉 no goals
  -/


open Algebra in
/-- If `μ` is a primitive `n`th root of unity in `R` and `k < n`, then `n` is divisible
by `(μ-1)^k` in `ℤ[μ] ⊆ R`. -/
lemma self_sub_one_pow_dvd_order {k n : ℕ} (hn : k < n) {μ : R} (hμ : IsPrimitiveRoot μ n) :
    ∃ z ∈ adjoin ℤ {μ}, n = z * (μ - 1) ^ k := by
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    k n : Nat
    hn : LT.lt k n
    μ : R
    hμ : IsPrimitiveRoot μ n
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  let n' + 1 := n
  /-
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    k n : Nat
    μ : R
    n' : Nat
    hn : LT.lt k (HAdd.hAdd n' 1)
    hμ : IsPrimitiveRoot μ (HAdd.hAdd n' 1)
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le' (Nat.le_of_lt_succ hn)
  have hdvd k : ∃ z ∈ adjoin ℤ {μ}, μ ^ k - 1 = z * (μ - 1) := by
    refine ⟨(Finset.range k).sum (μ ^ ·), ?_, (geom_sum_mul μ k).symm⟩
    exact Subalgebra.sum_mem _ fun m _ ↦ Subalgebra.pow_mem _ (self_mem_adjoin_singleton _ μ) _
  /-
    case intro
    R : Type u_1
    inst✝¹ : CommRing R
    inst✝ : IsDomain R
    k n : Nat
    μ : R
    m : Nat
    hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
    hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
    hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
    ⊢ Exists fun z => And (Membership.mem (Algebra.adjoin Int (Singleton.singleton …
  -/
  let Z k := Classical.choose <| hdvd k
  have Zdef k : Z k ∈ adjoin ℤ {μ} ∧ μ ^ k - 1 = Z k * (μ - 1) :=
    Classical.choose_spec <| hdvd k
  refine ⟨(-1) ^ (m + k) * (∏ j ∈ range k, Z (j + 1)) * ∏ j ∈ Ico k (m + k), (μ ^ (j + 1) - 1),
    ?_, ?_⟩
    /-
      case intro.refine_1
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      k n : Nat
      μ : R
      m : Nat
      hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
      hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
      hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
      Z : Nat → R := fun k => Classical.choose ⋯
      Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
      ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) (HMul.hMul (HMul …
    -/
  · apply Subalgebra.mul_mem
      /-
        case intro.refine_1.hx
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        k n : Nat
        μ : R
        m : Nat
        hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
        hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
        hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
        Z : Nat → R := fun k => Classical.choose ⋯
        Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
        ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) (HMul.hMul (HPow …
      -/
    · apply Subalgebra.mul_mem
        /-
          case intro.refine_1.hx.hx
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          k n : Nat
          μ : R
          m : Nat
          hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
          hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
          hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
          Z : Nat → R := fun k => Classical.choose ⋯
          Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
          ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) (HPow.hPow (-1)  …
        -/
      · exact Subalgebra.pow_mem _ (Subalgebra.neg_mem _ <| Subalgebra.one_mem _) _
        /-
          🎉 no goals
        -/
        /-
          case intro.refine_1.hx.hy
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          k n : Nat
          μ : R
          m : Nat
          hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
          hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
          hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
          Z : Nat → R := fun k => Classical.choose ⋯
          Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
          ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) ((Finset.range k …
        -/
      · exact Subalgebra.prod_mem _ fun _ _ ↦ (Zdef _).1
        /-
          🎉 no goals
        -/
      /-
        case intro.refine_1.hy
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        k n : Nat
        μ : R
        m : Nat
        hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
        hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
        hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
        Z : Nat → R := fun k => Classical.choose ⋯
        Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
        ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) ((Finset.Ico k ( …
      -/
    · refine Subalgebra.prod_mem _ fun _ _ ↦ ?_
      /-
        case intro.refine_1.hy
        R : Type u_1
        inst✝¹ : CommRing R
        inst✝ : IsDomain R
        k n : Nat
        μ : R
        m : Nat
        hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
        hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
        hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
        Z : Nat → R := fun k => Classical.choose ⋯
        Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
        x✝¹ : Nat
        x✝ : Membership.mem (Finset.Ico k (HAdd.hAdd m k)) x✝¹
        ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) (HSub.hSub (HPow …
      -/
      apply Subalgebra.sub_mem
        /-
          case intro.refine_1.hy.hx
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          k n : Nat
          μ : R
          m : Nat
          hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
          hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
          hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
          Z : Nat → R := fun k => Classical.choose ⋯
          Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
          x✝¹ : Nat
          x✝ : Membership.mem (Finset.Ico k (HAdd.hAdd m k)) x✝¹
          ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) (HPow.hPow μ (HA …
        -/
      · exact Subalgebra.pow_mem _ (self_mem_adjoin_singleton ℤ μ) _
        /-
          🎉 no goals
        -/
        /-
          case intro.refine_1.hy.hy
          R : Type u_1
          inst✝¹ : CommRing R
          inst✝ : IsDomain R
          k n : Nat
          μ : R
          m : Nat
          hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
          hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
          hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
          Z : Nat → R := fun k => Classical.choose ⋯
          Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
          x✝¹ : Nat
          x✝ : Membership.mem (Finset.Ico k (HAdd.hAdd m k)) x✝¹
          ⊢ Membership.mem (Algebra.adjoin Int (Singleton.singleton μ)) 1
        -/
      · exact Subalgebra.one_mem _
        /-
          🎉 no goals
        -/
    /-
      case intro.refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      k n : Nat
      μ : R
      m : Nat
      hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
      hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
      hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
      Z : Nat → R := fun k => Classical.choose ⋯
      Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
      ⊢ Eq (↑(HAdd.hAdd (HAdd.hAdd m k) 1)) (HMul.hMul (HMul.hMul (HMul.hMul (HPow.h …
    -/
  · push_cast
    /-
      case intro.refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      k n : Nat
      μ : R
      m : Nat
      hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
      hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
      hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
      Z : Nat → R := fun k => Classical.choose ⋯
      Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd ↑m ↑k) 1) (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hP …
    -/
    have := Nat.cast_add (R := R) m k ▸ hμ.prod_pow_sub_one_eq_order
    /-
      case intro.refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      k n : Nat
      μ : R
      m : Nat
      hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
      hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
      hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
      Z : Nat → R := fun k => Classical.choose ⋯
      Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
      this : Eq (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd m k)) ((Finset.range (HAdd.hAd …
      ⊢ Eq (HAdd.hAdd (HAdd.hAdd ↑m ↑k) 1) (HMul.hMul (HMul.hMul (HMul.hMul (HPow.hP …
    -/
    rw [← this, mul_assoc, mul_assoc]
    /-
      case intro.refine_2
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      k n : Nat
      μ : R
      m : Nat
      hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
      hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
      hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
      Z : Nat → R := fun k => Classical.choose ⋯
      Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
      this : Eq (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd m k)) ((Finset.range (HAdd.hAd …
      ⊢ Eq (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd m k)) ((Finset.range (HAdd.hAdd m k …
    -/
    congr 1
    /-
      case intro.refine_2.e_a
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      k n : Nat
      μ : R
      m : Nat
      hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
      hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
      hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
      Z : Nat → R := fun k => Classical.choose ⋯
      Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
      this : Eq (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd m k)) ((Finset.range (HAdd.hAd …
      ⊢ Eq ((Finset.range (HAdd.hAdd m k)).prod fun k => HSub.hSub (HPow.hPow μ (HAd …
    -/
    conv => enter [2, 2, 2]; rw [← card_range k]
    rw [← prod_range_mul_prod_Ico _ (Nat.le_add_left k m), mul_comm _ (_ ^ #_), ← mul_assoc,
      prod_mul_pow_card]
    /-
      case intro.refine_2.e_a
      R : Type u_1
      inst✝¹ : CommRing R
      inst✝ : IsDomain R
      k n : Nat
      μ : R
      m : Nat
      hn : LT.lt k (HAdd.hAdd (HAdd.hAdd m k) 1)
      hμ : IsPrimitiveRoot μ (HAdd.hAdd (HAdd.hAdd m k) 1)
      hdvd : ∀ (k : Nat), Exists fun z => And (Membership.mem (Algebra.adjoin Int (S …
      Z : Nat → R := fun k => Classical.choose ⋯
      Zdef : ∀ (k : Nat), And (Membership.mem (Algebra.adjoin Int (Singleton.singlet …
      this : Eq (HMul.hMul (HPow.hPow (-1) (HAdd.hAdd m k)) ((Finset.range (HAdd.hAd …
      ⊢ Eq (HMul.hMul ((Finset.range k).prod fun k => HSub.hSub (HPow.hPow μ (HAdd.h …
    -/
    conv => enter [2, 1, 2, j]; rw [← (Zdef _).2]
    /-
      🎉 no goals
    -/


