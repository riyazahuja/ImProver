/-- `n.factorization` is the finitely supported function `ℕ →₀ ℕ`
 mapping each prime factor of `n` to its multiplicity in `n`. -/
def factorization (n : ℕ) : ℕ →₀ ℕ where
  support := n.primeFactors
  toFun p := if p.Prime then padicValNat p n else 0
                          /-
                            a b m n✝ p n : Nat
                            ⊢ ∀ (a : Nat), Iff (Membership.mem n.primeFactors a) (Ne ((fun p => ite (Nat.P …
                          -/
  mem_support_toFun := by simp [not_or]; aesop
                                         /-
                                           🎉 no goals
                                         -/


/-- The support of `n.factorization` is exactly `n.primeFactors`. -/
@[simp] lemma support_factorization (n : ℕ) : (factorization n).support = n.primeFactors := rfl


theorem factorization_def (n : ℕ) {p : ℕ} (pp : p.Prime) : n.factorization p = padicValNat p n := by
  /-
    n p : Nat
    pp : Nat.Prime p
    ⊢ Eq (n.factorization p) (padicValNat p n)
  -/
  simpa [factorization] using absurd pp
  /-
    🎉 no goals
  -/


/-- We can write both `n.factorization p` and `n.factors.count p` to represent the power
of `p` in the factorization of `n`: we declare the former to be the simp-normal form. -/
@[simp]
theorem primeFactorsList_count_eq {n p : ℕ} : n.primeFactorsList.count p = n.factorization p := by
  /-
    n p : Nat
    ⊢ Eq (List.count p n.primeFactorsList) (n.factorization p)
  -/
  rcases n.eq_zero_or_pos with (rfl | hn0)
    /-
      case inl
      p : Nat
      ⊢ Eq (List.count p (Nat.primeFactorsList 0)) ((Nat.factorization 0) p)
    -/
  · simp [factorization, count]
    /-
      🎉 no goals
    -/
  if pp : p.Prime then ?_ else
    rw [count_eq_zero_of_not_mem (mt prime_of_mem_primeFactorsList pp)]
    simp [factorization, pp]
  /-
    case inr
    n p : Nat
    hn0 : GT.gt n 0
    pp : Nat.Prime p
    ⊢ Eq (List.count p n.primeFactorsList) (n.factorization p)
  -/
  simp only [factorization_def _ pp]
  /-
    case inr
    n p : Nat
    hn0 : GT.gt n 0
    pp : Nat.Prime p
    ⊢ Eq (List.count p n.primeFactorsList) (padicValNat p n)
  -/
  apply _root_.le_antisymm
    /-
      case inr.a
      n p : Nat
      hn0 : GT.gt n 0
      pp : Nat.Prime p
      ⊢ LE.le (List.count p n.primeFactorsList) (padicValNat p n)
    -/
  · rw [le_padicValNat_iff_replicate_subperm_primeFactorsList pp hn0.ne']
    /-
      case inr.a
      n p : Nat
      hn0 : GT.gt n 0
      pp : Nat.Prime p
      ⊢ (List.replicate (List.count p n.primeFactorsList) p).Subperm n.primeFactorsL …
    -/
    exact List.le_count_iff_replicate_sublist.mp le_rfl |>.subperm
    /-
      🎉 no goals
    -/
  · rw [← Nat.lt_add_one_iff, lt_iff_not_ge, ge_iff_le,
      le_padicValNat_iff_replicate_subperm_primeFactorsList pp hn0.ne']
    /-
      case inr.a
      n p : Nat
      hn0 : GT.gt n 0
      pp : Nat.Prime p
      ⊢ Not ((List.replicate (HAdd.hAdd (List.count p n.primeFactorsList) 1) p).Subp …
    -/
    intro h
    /-
      case inr.a
      n p : Nat
      hn0 : GT.gt n 0
      pp : Nat.Prime p
      h : (List.replicate (HAdd.hAdd (List.count p n.primeFactorsList) 1) p).Subperm …
      ⊢ False
    -/
    have := h.count_le p
    /-
      case inr.a
      n p : Nat
      hn0 : GT.gt n 0
      pp : Nat.Prime p
      h : (List.replicate (HAdd.hAdd (List.count p n.primeFactorsList) 1) p).Subperm …
      this : LE.le (List.count p (List.replicate (HAdd.hAdd (List.count p n.primeFac …
      ⊢ False
    -/
    simp at this
    /-
      🎉 no goals
    -/


theorem factorization_eq_primeFactorsList_multiset (n : ℕ) :
    n.factorization = Multiset.toFinsupp (n.primeFactorsList : Multiset ℕ) := by
  /-
    n : Nat
    ⊢ Eq n.factorization (Multiset.toFinsupp ↑n.primeFactorsList)
  -/
  ext p
  /-
    case h
    n p : Nat
    ⊢ Eq (n.factorization p) ((Multiset.toFinsupp ↑n.primeFactorsList) p)
  -/
  simp
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-07-16")] alias factors_count_eq := primeFactorsList_count_eq

@[deprecated (since := "2024-07-16")]
alias factorization_eq_factors_multiset := factorization_eq_primeFactorsList_multiset


theorem Prime.factorization_pos_of_dvd {n p : ℕ} (hp : p.Prime) (hn : n ≠ 0) (h : p ∣ n) :
    0 < n.factorization p := by
    /-
      n p : Nat
      hp : Nat.Prime p
      hn : Ne n 0
      h : Dvd.dvd p n
      ⊢ LT.lt 0 (n.factorization p)
    -/
    rwa [← primeFactorsList_count_eq, count_pos_iff, mem_primeFactorsList_iff_dvd hn hp]
    /-
      🎉 no goals
    -/


theorem multiplicity_eq_factorization {n p : ℕ} (pp : p.Prime) (hn : n ≠ 0) :
    multiplicity p n = n.factorization p := by
  /-
    n p : Nat
    pp : Nat.Prime p
    hn : Ne n 0
    ⊢ Eq (multiplicity p n) (n.factorization p)
  -/
  simp [factorization, pp, padicValNat_def' pp.ne_one hn.bot_lt]
  /-
    🎉 no goals
  -/


@[simp]
theorem factorization_prod_pow_eq_self {n : ℕ} (hn : n ≠ 0) : n.factorization.prod (· ^ ·) = n := by
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq (n.factorization.prod fun x1 x2 => HPow.hPow x1 x2) n
  -/
  rw [factorization_eq_primeFactorsList_multiset n]
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq ((Multiset.toFinsupp ↑n.primeFactorsList).prod fun x1 x2 => HPow.hPow x1  …
  -/
  simp only [← prod_toMultiset, factorization, Multiset.prod_coe, Multiset.toFinsupp_toMultiset]
  /-
    n : Nat
    hn : Ne n 0
    ⊢ Eq n.primeFactorsList.prod n
  -/
  exact prod_primeFactorsList hn
  /-
    🎉 no goals
  -/


theorem eq_of_factorization_eq {a b : ℕ} (ha : a ≠ 0) (hb : b ≠ 0)
    (h : ∀ p : ℕ, a.factorization p = b.factorization p) : a = b :=
  eq_of_perm_primeFactorsList ha hb
        /-
          a b : Nat
          ha : Ne a 0
          hb : Ne b 0
          h : ∀ (p : Nat), Eq (a.factorization p) (b.factorization p)
          ⊢ a.primeFactorsList.Perm b.primeFactorsList
        -/
    (by simpa only [List.perm_iff_count, primeFactorsList_count_eq] using h)
        /-
          🎉 no goals
        -/



/-- Every nonzero natural number has a unique prime factorization -/
theorem factorization_inj : Set.InjOn factorization { x : ℕ | x ≠ 0 } := fun a ha b hb h =>
                                           /-
                                             a : Nat
                                             ha : Membership.mem (setOf fun x => Ne x 0) a
                                             b : Nat
                                             hb : Membership.mem (setOf fun x => Ne x 0) b
                                             h : Eq a.factorization b.factorization
                                             p : Nat
                                             ⊢ Eq (a.factorization p) (b.factorization p)
                                           -/
  eq_of_factorization_eq ha hb fun p => by simp [h]
                                           /-
                                             🎉 no goals
                                           -/


@[simp]
                                                       /-
                                                         ⊢ Eq (Nat.factorization 0) 0
                                                       -/
theorem factorization_zero : factorization 0 = 0 := by ext; simp [factorization]
                                                            /-
                                                              🎉 no goals
                                                            -/


@[simp]
                                                      /-
                                                        ⊢ Eq (Nat.factorization 1) 0
                                                      -/
theorem factorization_one : factorization 1 = 0 := by ext; simp [factorization]
                                                           /-
                                                             🎉 no goals
                                                           -/


theorem factorization_eq_zero_iff (n p : ℕ) :
    n.factorization p = 0 ↔ ¬p.Prime ∨ ¬p ∣ n ∨ n = 0 := by
  /-
    n p : Nat
    ⊢ Iff (Eq (n.factorization p) 0) (Or (Not (Nat.Prime p)) (Or (Not (Dvd.dvd p n …
  -/
  simp_rw [← not_mem_support_iff, support_factorization, mem_primeFactors, not_and_or, not_ne_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem factorization_eq_zero_of_non_prime (n : ℕ) {p : ℕ} (hp : ¬p.Prime) :
                                /-
                                  n p : Nat
                                  hp : Not (Nat.Prime p)
                                  ⊢ Eq (n.factorization p) 0
                                -/
    n.factorization p = 0 := by simp [factorization_eq_zero_iff, hp]
                                /-
                                  🎉 no goals
                                -/


@[simp]
theorem factorization_zero_right (n : ℕ) : n.factorization 0 = 0 :=
  factorization_eq_zero_of_non_prime _ not_prime_zero


theorem factorization_eq_zero_of_not_dvd {n p : ℕ} (h : ¬p ∣ n) : n.factorization p = 0 := by
  /-
    n p : Nat
    h : Not (Dvd.dvd p n)
    ⊢ Eq (n.factorization p) 0
  -/
  simp [factorization_eq_zero_iff, h]
  /-
    🎉 no goals
  -/


theorem factorization_eq_zero_of_remainder {p r : ℕ} (i : ℕ) (hr : ¬p ∣ r) :
    (p * i + r).factorization p = 0 := by
  /-
    p r i : Nat
    hr : Not (Dvd.dvd p r)
    ⊢ Eq ((HAdd.hAdd (HMul.hMul p i) r).factorization p) 0
  -/
  apply factorization_eq_zero_of_not_dvd
  /-
    case h
    p r i : Nat
    hr : Not (Dvd.dvd p r)
    ⊢ Not (Dvd.dvd p (HAdd.hAdd (HMul.hMul p i) r))
  -/
  rwa [← Nat.dvd_add_iff_right (Dvd.intro i rfl)]
  /-
    🎉 no goals
  -/


/-- For nonzero `a` and `b`, the power of `p` in `a * b` is the sum of the powers in `a` and `b` -/
@[simp]
theorem factorization_mul {a b : ℕ} (ha : a ≠ 0) (hb : b ≠ 0) :
    (a * b).factorization = a.factorization + b.factorization := by
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ Eq (HMul.hMul a b).factorization (HAdd.hAdd a.factorization b.factorization)
  -/
  ext p
  simp only [add_apply, ← primeFactorsList_count_eq,
    perm_iff_count.mp (perm_primeFactorsList_mul ha hb) p, count_append]


theorem factorization_le_iff_dvd {d n : ℕ} (hd : d ≠ 0) (hn : n ≠ 0) :
    d.factorization ≤ n.factorization ↔ d ∣ n := by
  /-
    d n : Nat
    hd : Ne d 0
    hn : Ne n 0
    ⊢ Iff (LE.le d.factorization n.factorization) (Dvd.dvd d n)
  -/
  constructor
    /-
      case mp
      d n : Nat
      hd : Ne d 0
      hn : Ne n 0
      ⊢ LE.le d.factorization n.factorization → Dvd.dvd d n
    -/
  · intro hdn
    /-
      case mp
      d n : Nat
      hd : Ne d 0
      hn : Ne n 0
      hdn : LE.le d.factorization n.factorization
      ⊢ Dvd.dvd d n
    -/
    set K := n.factorization - d.factorization with hK
    /-
      case mp
      d n : Nat
      hd : Ne d 0
      hn : Ne n 0
      hdn : LE.le d.factorization n.factorization
      K : Finsupp Nat Nat := HSub.hSub n.factorization d.factorization
      hK : Eq K (HSub.hSub n.factorization d.factorization)
      ⊢ Dvd.dvd d n
    -/
    use K.prod (· ^ ·)
    rw [← factorization_prod_pow_eq_self hn, ← factorization_prod_pow_eq_self hd,
        ← Finsupp.prod_add_index' pow_zero pow_add, hK, add_tsub_cancel_of_le hdn]
    /-
      case mpr
      d n : Nat
      hd : Ne d 0
      hn : Ne n 0
      ⊢ Dvd.dvd d n → LE.le d.factorization n.factorization
    -/
  · rintro ⟨c, rfl⟩
    /-
      case mpr.intro
      d : Nat
      hd : Ne d 0
      c : Nat
      hn : Ne (HMul.hMul d c) 0
      ⊢ LE.le d.factorization (HMul.hMul d c).factorization
    -/
    rw [factorization_mul hd (right_ne_zero_of_mul hn)]
    /-
      case mpr.intro
      d : Nat
      hd : Ne d 0
      c : Nat
      hn : Ne (HMul.hMul d c) 0
      ⊢ LE.le d.factorization (HAdd.hAdd d.factorization c.factorization)
    -/
    simp
    /-
      🎉 no goals
    -/


/-- For any `p : ℕ` and any function `g : α → ℕ` that's non-zero on `S : Finset α`,
the power of `p` in `S.prod g` equals the sum over `x ∈ S` of the powers of `p` in `g x`.
Generalises `factorization_mul`, which is the special case where `#S = 2` and `g = id`. -/
theorem factorization_prod {α : Type*} {S : Finset α} {g : α → ℕ} (hS : ∀ x ∈ S, g x ≠ 0) :
    (S.prod g).factorization = S.sum fun x => (g x).factorization := by
  classical
    ext p
    refine Finset.induction_on' S ?_ ?_
    · simp
    · intro x T hxS hTS hxT IH
      have hT : T.prod g ≠ 0 := prod_ne_zero_iff.mpr fun x hx => hS x (hTS hx)
      simp [prod_insert hxT, sum_insert hxT, IH, factorization_mul (hS x hxS) hT]


/-- For any `p`, the power of `p` in `n^k` is `k` times the power in `n` -/
@[simp]
theorem factorization_pow (n k : ℕ) : factorization (n ^ k) = k • n.factorization := by
  /-
    n k : Nat
    ⊢ Eq (HPow.hPow n k).factorization (HSMul.hSMul k n.factorization)
  -/
  induction' k with k ih; · simp
                            /-
                              🎉 no goals
                            -/
  /-
    case succ
    n k : Nat
    ih : Eq (HPow.hPow n k).factorization (HSMul.hSMul k n.factorization)
    ⊢ Eq (HPow.hPow n (HAdd.hAdd k 1)).factorization (HSMul.hSMul (HAdd.hAdd k 1)  …
  -/
  rcases eq_or_ne n 0 with (rfl | hn)
    /-
      case succ.inl
      k : Nat
      ih : Eq (HPow.hPow 0 k).factorization (HSMul.hSMul k (Nat.factorization 0))
      ⊢ Eq (HPow.hPow 0 (HAdd.hAdd k 1)).factorization (HSMul.hSMul (HAdd.hAdd k 1)  …
    -/
  · simp
    /-
      🎉 no goals
    -/
  rw [Nat.pow_succ, mul_comm, factorization_mul hn (pow_ne_zero _ hn), ih,
    add_smul, one_smul, add_comm]


/-- The only prime factor of prime `p` is `p` itself, with multiplicity `1` -/
@[simp]
protected theorem Prime.factorization {p : ℕ} (hp : Prime p) : p.factorization = single p 1 := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq p.factorization (Finsupp.single p 1)
  -/
  ext q
  rw [← primeFactorsList_count_eq, primeFactorsList_prime hp, single_apply, count_singleton',
                          /-
                            case h.h_t
                            p : Nat
                            hp : Nat.Prime p
                            q : Nat
                            ⊢ Eq 1 ?m.33592
                          -/
                          /-
                            🎉 no goals
                          -/
    if_congr eq_comm] <;> rfl
                          /-
                            🎉 no goals
                          -/


/-- For prime `p` the only prime factor of `p^k` is `p` with multiplicity `k` -/
theorem Prime.factorization_pow {p k : ℕ} (hp : Prime p) : (p ^ k).factorization = single p k := by
  /-
    p k : Nat
    hp : Nat.Prime p
    ⊢ Eq (HPow.hPow p k).factorization (Finsupp.single p k)
  -/
  simp [hp]
  /-
    🎉 no goals
  -/


theorem pow_succ_factorization_not_dvd {n p : ℕ} (hn : n ≠ 0) (hp : p.Prime) :
    ¬p ^ (n.factorization p + 1) ∣ n := by
  /-
    n p : Nat
    hn : Ne n 0
    hp : Nat.Prime p
    ⊢ Not (Dvd.dvd (HPow.hPow p (HAdd.hAdd (n.factorization p) 1)) n)
  -/
  intro h
  /-
    n p : Nat
    hn : Ne n 0
    hp : Nat.Prime p
    h : Dvd.dvd (HPow.hPow p (HAdd.hAdd (n.factorization p) 1)) n
    ⊢ False
  -/
  rw [← factorization_le_iff_dvd (pow_pos hp.pos _).ne' hn] at h
  /-
    n p : Nat
    hn : Ne n 0
    hp : Nat.Prime p
    h : LE.le (HPow.hPow p (HAdd.hAdd (n.factorization p) 1)).factorization n.fact …
    ⊢ False
  -/
  simpa [hp.factorization] using h p
  /-
    🎉 no goals
  -/


/-- Any Finsupp `f : ℕ →₀ ℕ` whose support is in the primes is equal to the factorization of
the product `∏ (a : ℕ) ∈ f.support, a ^ f a`. -/
theorem prod_pow_factorization_eq_self {f : ℕ →₀ ℕ} (hf : ∀ p : ℕ, p ∈ f.support → Prime p) :
    (f.prod (· ^ ·)).factorization = f := by
  have h : ∀ x : ℕ, x ∈ f.support → x ^ f x ≠ 0 := fun p hp =>
    pow_ne_zero _ (Prime.ne_zero (hf p hp))
  /-
    f : Finsupp Nat Nat
    hf : ∀ (p : Nat), Membership.mem f.support p → Nat.Prime p
    h : ∀ (x : Nat), Membership.mem f.support x → Ne (HPow.hPow x (f x)) 0
    ⊢ Eq (f.prod fun x1 x2 => HPow.hPow x1 x2).factorization f
  -/
  simp only [Finsupp.prod, factorization_prod h]
  conv =>
    rhs
    rw [(sum_single f).symm]
  /-
    f : Finsupp Nat Nat
    hf : ∀ (p : Nat), Membership.mem f.support p → Nat.Prime p
    h : ∀ (x : Nat), Membership.mem f.support x → Ne (HPow.hPow x (f x)) 0
    ⊢ Eq (f.support.sum fun x => (HPow.hPow x (f x)).factorization) (f.sum Finsupp …
  -/
  exact sum_congr rfl fun p hp => Prime.factorization_pow (hf p hp)
  /-
    🎉 no goals
  -/


/-- The equiv between `ℕ+` and `ℕ →₀ ℕ` with support in the primes. -/
def factorizationEquiv : ℕ+ ≃ { f : ℕ →₀ ℕ | ∀ p ∈ f.support, Prime p } where
  toFun := fun ⟨n, _⟩ => ⟨n.factorization, fun _ => prime_of_mem_primeFactors⟩
  invFun := fun ⟨f, hf⟩ =>
    ⟨f.prod _, prod_pow_pos_of_zero_not_mem_support fun H => not_prime_zero (hf 0 H)⟩
  left_inv := fun ⟨_, hx⟩ => Subtype.ext <| factorization_prod_pow_eq_self hx.ne.symm
  right_inv := fun ⟨_, hf⟩ => Subtype.ext <| prod_pow_factorization_eq_self hf


/-- For coprime `a` and `b`, the power of `p` in `a * b` is the sum of the powers in `a` and `b` -/
theorem factorization_mul_apply_of_coprime {p a b : ℕ} (hab : Coprime a b) :
    (a * b).factorization p = a.factorization p + b.factorization p := by
  simp only [← primeFactorsList_count_eq,
    perm_iff_count.mp (perm_primeFactorsList_mul_of_coprime hab), count_append]


/-- For coprime `a` and `b`, the power of `p` in `a * b` is the sum of the powers in `a` and `b` -/
theorem factorization_mul_of_coprime {a b : ℕ} (hab : Coprime a b) :
    (a * b).factorization = a.factorization + b.factorization := by
  /-
    a b : Nat
    hab : a.Coprime b
    ⊢ Eq (HMul.hMul a b).factorization (HAdd.hAdd a.factorization b.factorization)
  -/
  ext q
  /-
    case h
    a b : Nat
    hab : a.Coprime b
    q : Nat
    ⊢ Eq ((HMul.hMul a b).factorization q) ((HAdd.hAdd a.factorization b.factoriza …
  -/
  rw [Finsupp.add_apply, factorization_mul_apply_of_coprime hab]
  /-
    🎉 no goals
  -/


/-- We introduce the notations `ordProj[p] n` for the largest power of the prime `p` that
divides `n` and `ordCompl[p] n` for the complementary part. The `ord` naming comes from
the $p$-adic order/valuation of a number, and `proj` and `compl` are for the projection and
complementary projection. The term `n.factorization p` is the $p$-adic order itself.
For example, `ordProj[2] n` is the even part of `n` and `ordCompl[2] n` is the odd part. -/
notation "ordProj[" p "] " n:arg => p ^ Nat.factorization n p


@[inherit_doc «termOrdProj[_]_»]
notation "ordCompl[" p "] " n:arg => n / ordProj[p] n


theorem ordProj_dvd (n p : ℕ) : ordProj[p] n ∣ n := by
  /-
    n p : Nat
    ⊢ Dvd.dvd (HPow.hPow p (n.factorization p)) n
  -/
  if hp : p.Prime then ?_ else simp [hp]
  /-
    n p : Nat
    hp : Nat.Prime p
    ⊢ Dvd.dvd (HPow.hPow p (n.factorization p)) n
  -/
  rw [← primeFactorsList_count_eq]
  /-
    n p : Nat
    hp : Nat.Prime p
    ⊢ Dvd.dvd (HPow.hPow p (List.count p n.primeFactorsList)) n
  -/
  apply dvd_of_primeFactorsList_subperm (pow_ne_zero _ hp.ne_zero)
  /-
    n p : Nat
    hp : Nat.Prime p
    ⊢ (HPow.hPow p (List.count p n.primeFactorsList)).primeFactorsList.Subperm n.p …
  -/
  rw [hp.primeFactorsList_pow, List.subperm_ext_iff]
  /-
    n p : Nat
    hp : Nat.Prime p
    ⊢ ∀ (x : Nat), Membership.mem (List.replicate (List.count p n.primeFactorsList …
  -/
  intro q hq
  /-
    n p : Nat
    hp : Nat.Prime p
    q : Nat
    hq : Membership.mem (List.replicate (List.count p n.primeFactorsList) p) q
    ⊢ LE.le (List.count q (List.replicate (List.count p n.primeFactorsList) p)) (L …
  -/
  simp [List.eq_of_mem_replicate hq]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-10-24")] alias ord_proj_dvd := ordProj_dvd


/-- If `a = ∏ pᵢ ^ nᵢ` and `b = ∏ pᵢ ^ mᵢ`, then `factorizationLCMLeft = ∏ pᵢ ^ kᵢ`, where
`kᵢ = nᵢ` if `mᵢ ≤ nᵢ` and `0` otherwise. Note that the product is over the divisors of `lcm a b`,
so if one of `a` or `b` is `0` then the result is `1`. -/
def factorizationLCMLeft (a b : ℕ) : ℕ :=
  (Nat.lcm a b).factorization.prod fun p n ↦
    if b.factorization p ≤ a.factorization p then p ^ n else 1


/-- If `a = ∏ pᵢ ^ nᵢ` and `b = ∏ pᵢ ^ mᵢ`, then `factorizationLCMRight = ∏ pᵢ ^ kᵢ`, where
`kᵢ = mᵢ` if `nᵢ < mᵢ` and `0` otherwise. Note that the product is over the divisors of `lcm a b`,
so if one of `a` or `b` is `0` then the result is `1`.

Note that `factorizationLCMRight a b` is *not* `factorizationLCMLeft b a`: the difference is
that in `factorizationLCMLeft a b` there are the primes whose exponent in `a` is bigger or equal
than the exponent in `b`, while in `factorizationLCMRight a b` there are the primes whose
exponent in `b` is strictly bigger than in `a`. For example `factorizationLCMLeft 2 2 = 2`, but
`factorizationLCMRight 2 2 = 1`. -/
def factorizationLCMRight (a b : ℕ) :=
  (Nat.lcm a b).factorization.prod fun p n ↦
    if b.factorization p ≤ a.factorization p then 1 else p ^ n


