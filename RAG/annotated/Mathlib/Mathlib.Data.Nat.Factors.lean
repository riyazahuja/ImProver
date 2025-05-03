/-- `primeFactorsList n` is the prime factorization of `n`, listed in increasing order. -/
def primeFactorsList : ℕ → List ℕ
  | 0 => []
  | 1 => []
  | k + 2 =>
    let m := minFac (k + 2)
    m :: primeFactorsList ((k + 2) / m)
/-
  k : Nat
  m : Nat := (HAdd.hAdd k 2).minFac
  ⊢ LT.lt (HDiv.hDiv (HAdd.hAdd k 2) (HAdd.hAdd k 2).minFac) k.succ.succ
-/
decreasing_by exact factors_lemma
/-
  🎉 no goals
-/


@[deprecated (since := "2024-06-14")] alias factors := primeFactorsList


@[simp]
                                                              /-
                                                                ⊢ Eq (Nat.primeFactorsList 0) List.nil
                                                              -/
theorem primeFactorsList_zero : primeFactorsList 0 = [] := by rw [primeFactorsList]
                                                              /-
                                                                🎉 no goals
                                                              -/


@[simp]
                                                             /-
                                                               ⊢ Eq (Nat.primeFactorsList 1) List.nil
                                                             -/
theorem primeFactorsList_one : primeFactorsList 1 = [] := by rw [primeFactorsList]
                                                             /-
                                                               🎉 no goals
                                                             -/


@[simp]
                                                              /-
                                                                ⊢ Eq (Nat.primeFactorsList 2) (List.cons 2 List.nil)
                                                              -/
theorem primeFactorsList_two : primeFactorsList 2 = [2] := by simp [primeFactorsList]
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem prime_of_mem_primeFactorsList {n : ℕ} : ∀ {p : ℕ}, p ∈ primeFactorsList n → Prime p := by
  match n with
  | 0 => simp
  | 1 => simp
  | k + 2 =>
      intro p h
      let m := minFac (k + 2)
      have : (k + 2) / m < (k + 2) := factors_lemma
      have h₁ : p = m ∨ p ∈ primeFactorsList ((k + 2) / m) :=
        List.mem_cons.1 (by rwa [primeFactorsList] at h)
      exact Or.casesOn h₁ (fun h₂ => h₂.symm ▸ minFac_prime (by simp)) prime_of_mem_primeFactorsList


theorem pos_of_mem_primeFactorsList {n p : ℕ} (h : p ∈ primeFactorsList n) : 0 < p :=
  Prime.pos (prime_of_mem_primeFactorsList h)


theorem prod_primeFactorsList : ∀ {n}, n ≠ 0 → List.prod (primeFactorsList n) = n
            /-
              ⊢ Ne 0 0 → Eq (Nat.primeFactorsList 0).prod 0
            -/
  | 0 => by simp
            /-
              🎉 no goals
            -/
            /-
              ⊢ Ne 1 0 → Eq (Nat.primeFactorsList 1).prod 1
            -/
  | 1 => by simp
            /-
              🎉 no goals
            -/
  | k + 2 => fun _ =>
    let m := minFac (k + 2)
    have : (k + 2) / m < (k + 2) := factors_lemma
    show (primeFactorsList (k + 2)).prod = (k + 2) by
      have h₁ : (k + 2) / m ≠ 0 := fun h => by
        have : (k + 2) = 0 * m := (Nat.div_eq_iff_eq_mul_left (minFac_pos _) (minFac_dvd _)).1 h
        rw [zero_mul] at this; exact (show k + 2 ≠ 0 by simp) this
      rw [primeFactorsList, List.prod_cons, prod_primeFactorsList h₁,
        Nat.mul_div_cancel' (minFac_dvd _)]


theorem primeFactorsList_prime {p : ℕ} (hp : Nat.Prime p) : p.primeFactorsList = [p] := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq p.primeFactorsList (List.cons p List.nil)
  -/
  have : p = p - 2 + 2 := Nat.eq_add_of_sub_eq hp.two_le rfl
  /-
    p : Nat
    hp : Nat.Prime p
    this : Eq p (HAdd.hAdd (HSub.hSub p 2) 2)
    ⊢ Eq p.primeFactorsList (List.cons p List.nil)
  -/
  rw [this, primeFactorsList]
  /-
    p : Nat
    hp : Nat.Prime p
    this : Eq p (HAdd.hAdd (HSub.hSub p 2) 2)
    ⊢ Eq (List.cons (HAdd.hAdd (HSub.hSub p 2) 2).minFac (HDiv.hDiv (HAdd.hAdd (HS …
  -/
  simp only [Eq.symm this]
  /-
    p : Nat
    hp : Nat.Prime p
    this : Eq p (HAdd.hAdd (HSub.hSub p 2) 2)
    ⊢ Eq (List.cons p.minFac (HDiv.hDiv p p.minFac).primeFactorsList) (List.cons p …
  -/
  have : Nat.minFac p = p := (Nat.prime_def_minFac.mp hp).2
  /-
    p : Nat
    hp : Nat.Prime p
    this✝ : Eq p (HAdd.hAdd (HSub.hSub p 2) 2)
    this : Eq p.minFac p
    ⊢ Eq (List.cons p.minFac (HDiv.hDiv p p.minFac).primeFactorsList) (List.cons p …
  -/
  simp only [this, primeFactorsList, Nat.div_self (Nat.Prime.pos hp)]
  /-
    🎉 no goals
  -/


theorem primeFactorsList_chain {n : ℕ} :
    ∀ {a}, (∀ p, Prime p → p ∣ n → a ≤ p) → List.Chain (· ≤ ·) a (primeFactorsList n) := by
  match n with
  | 0 => simp
  | 1 => simp
  | k + 2 =>
      intro a h
      let m := minFac (k + 2)
      have : (k + 2) / m < (k + 2) := factors_lemma
      rw [primeFactorsList]
      refine List.Chain.cons ((le_minFac.2 h).resolve_left (by simp)) (primeFactorsList_chain ?_)
      exact fun p pp d => minFac_le_of_dvd pp.two_le (d.trans <| div_dvd_of_dvd <| minFac_dvd _)


theorem primeFactorsList_chain_2 (n) : List.Chain (· ≤ ·) 2 (primeFactorsList n) :=
  primeFactorsList_chain fun _ pp _ => pp.two_le


theorem primeFactorsList_chain' (n) : List.Chain' (· ≤ ·) (primeFactorsList n) :=
  @List.Chain'.tail _ _ (_ :: _) (primeFactorsList_chain_2 _)


theorem primeFactorsList_sorted (n : ℕ) : List.Sorted (· ≤ ·) (primeFactorsList n) :=
  List.chain'_iff_pairwise.1 (primeFactorsList_chain' _)


/-- `primeFactorsList` can be constructed inductively by extracting `minFac`, for sufficiently
large `n`. -/
theorem primeFactorsList_add_two (n : ℕ) :
    primeFactorsList (n + 2) = minFac (n + 2) :: primeFactorsList ((n + 2) / minFac (n + 2)) := by
  /-
    n : Nat
    ⊢ Eq (HAdd.hAdd n 2).primeFactorsList (List.cons (HAdd.hAdd n 2).minFac (HDiv. …
  -/
  rw [primeFactorsList]
  /-
    🎉 no goals
  -/


@[simp]
theorem primeFactorsList_eq_nil (n : ℕ) : n.primeFactorsList = [] ↔ n = 0 ∨ n = 1 := by
  /-
    n : Nat
    ⊢ Iff (Eq n.primeFactorsList List.nil) (Or (Eq n 0) (Eq n 1))
  -/
  constructor <;> intro h
    /-
      case mp
      n : Nat
      h : Eq n.primeFactorsList List.nil
      ⊢ Or (Eq n 0) (Eq n 1)
    -/
  · rcases n with (_ | _ | n)
      /-
        case mp.zero
        h : Eq (Nat.primeFactorsList 0) List.nil
        ⊢ Or (Eq 0 0) (Eq 0 1)
      -/
    · exact Or.inl rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.succ.zero
        h : Eq (HAdd.hAdd 0 1).primeFactorsList List.nil
        ⊢ Or (Eq (HAdd.hAdd 0 1) 0) (Eq (HAdd.hAdd 0 1) 1)
      -/
    · exact Or.inr rfl
      /-
        🎉 no goals
      -/
      /-
        case mp.succ.succ
        n : Nat
        h : Eq (HAdd.hAdd (HAdd.hAdd n 1) 1).primeFactorsList List.nil
        ⊢ Or (Eq (HAdd.hAdd (HAdd.hAdd n 1) 1) 0) (Eq (HAdd.hAdd (HAdd.hAdd n 1) 1) 1)
      -/
    · rw [primeFactorsList] at h
      /-
        case mp.succ.succ
        n : Nat
        h : Eq (List.cons (HAdd.hAdd n 2).minFac (HDiv.hDiv (HAdd.hAdd n 2) (HAdd.hAdd …
        ⊢ Or (Eq (HAdd.hAdd (HAdd.hAdd n 1) 1) 0) (Eq (HAdd.hAdd (HAdd.hAdd n 1) 1) 1)
      -/
      injection h
      /-
        🎉 no goals
      -/
    /-
      case mpr
      n : Nat
      h : Or (Eq n 0) (Eq n 1)
      ⊢ Eq n.primeFactorsList List.nil
    -/
  · rcases h with (rfl | rfl)
      /-
        case mpr.inl
        ⊢ Eq (Nat.primeFactorsList 0) List.nil
      -/
    · exact primeFactorsList_zero
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        ⊢ Eq (Nat.primeFactorsList 1) List.nil
      -/
    · exact primeFactorsList_one
      /-
        🎉 no goals
      -/


open scoped List in
theorem eq_of_perm_primeFactorsList {a b : ℕ} (ha : a ≠ 0) (hb : b ≠ 0)
    (h : a.primeFactorsList ~ b.primeFactorsList) : a = b := by
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    h : a.primeFactorsList.Perm b.primeFactorsList
    ⊢ Eq a b
  -/
  simpa [prod_primeFactorsList ha, prod_primeFactorsList hb] using List.Perm.prod_eq h
  /-
    🎉 no goals
  -/


theorem mem_primeFactorsList_iff_dvd {n p : ℕ} (hn : n ≠ 0) (hp : Prime p) :
    p ∈ primeFactorsList n ↔ p ∣ n where
  mp h := prod_primeFactorsList hn ▸ List.dvd_prod h
  mpr h := mem_list_primes_of_dvd_prod (prime_iff.mp hp)
    (fun _ h ↦ prime_iff.mp (prime_of_mem_primeFactorsList h)) ((prod_primeFactorsList hn).symm ▸ h)


theorem dvd_of_mem_primeFactorsList {n p : ℕ} (h : p ∈ n.primeFactorsList) : p ∣ n := by
  /-
    n p : Nat
    h : Membership.mem n.primeFactorsList p
    ⊢ Dvd.dvd p n
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      p : Nat
      h : Membership.mem (Nat.primeFactorsList 0) p
      ⊢ Dvd.dvd p 0
    -/
  · exact dvd_zero p
    /-
      🎉 no goals
    -/
    /-
      case inr
      n p : Nat
      h : Membership.mem n.primeFactorsList p
      hn : GT.gt n 0
      ⊢ Dvd.dvd p n
    -/
  · rwa [← mem_primeFactorsList_iff_dvd hn.ne' (prime_of_mem_primeFactorsList h)]
    /-
      🎉 no goals
    -/


theorem mem_primeFactorsList {n p} (hn : n ≠ 0) : p ∈ primeFactorsList n ↔ Prime p ∧ p ∣ n :=
  ⟨fun h => ⟨prime_of_mem_primeFactorsList h, dvd_of_mem_primeFactorsList h⟩, fun ⟨hprime, hdvd⟩ =>
    (mem_primeFactorsList_iff_dvd hn hprime).mpr hdvd⟩


@[simp] lemma mem_primeFactorsList' {n p} : p ∈ n.primeFactorsList ↔ p.Prime ∧ p ∣ n ∧ n ≠ 0 := by
  /-
    n p : Nat
    ⊢ Iff (Membership.mem n.primeFactorsList p) (And (Nat.Prime p) (And (Dvd.dvd p …
  -/
              /-
                🎉 no goals
              -/
  cases n <;> simp [mem_primeFactorsList, *]
              /-
                🎉 no goals
              -/


theorem le_of_mem_primeFactorsList {n p : ℕ} (h : p ∈ n.primeFactorsList) : p ≤ n := by
  /-
    n p : Nat
    h : Membership.mem n.primeFactorsList p
    ⊢ LE.le p n
  -/
  rcases n.eq_zero_or_pos with (rfl | hn)
    /-
      case inl
      p : Nat
      h : Membership.mem (Nat.primeFactorsList 0) p
      ⊢ LE.le p 0
    -/
  · rw [primeFactorsList_zero] at h
    /-
      case inl
      p : Nat
      h : Membership.mem List.nil p
      ⊢ LE.le p 0
    -/
    cases h
    /-
      🎉 no goals
    -/
    /-
      case inr
      n p : Nat
      h : Membership.mem n.primeFactorsList p
      hn : GT.gt n 0
      ⊢ LE.le p n
    -/
  · exact le_of_dvd hn (dvd_of_mem_primeFactorsList h)
    /-
      🎉 no goals
    -/


/-- **Fundamental theorem of arithmetic**-/
theorem primeFactorsList_unique {n : ℕ} {l : List ℕ} (h₁ : prod l = n) (h₂ : ∀ p ∈ l, Prime p) :
    l ~ primeFactorsList n := by
  /-
    n : Nat
    l : List Nat
    h₁ : Eq l.prod n
    h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
    ⊢ l.Perm n.primeFactorsList
  -/
  refine perm_of_prod_eq_prod ?_ ?_ ?_
    /-
      case refine_1
      n : Nat
      l : List Nat
      h₁ : Eq l.prod n
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      ⊢ Eq l.prod n.primeFactorsList.prod
    -/
  · rw [h₁]
    /-
      case refine_1
      n : Nat
      l : List Nat
      h₁ : Eq l.prod n
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      ⊢ Eq n n.primeFactorsList.prod
    -/
    refine (prod_primeFactorsList ?_).symm
    /-
      case refine_1
      n : Nat
      l : List Nat
      h₁ : Eq l.prod n
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      ⊢ Ne n 0
    -/
    rintro rfl
    /-
      case refine_1
      l : List Nat
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      h₁ : Eq l.prod 0
      ⊢ False
    -/
    rw [prod_eq_zero_iff] at h₁
    /-
      case refine_1
      l : List Nat
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      h₁ : Membership.mem l 0
      ⊢ False
    -/
    exact Prime.ne_zero (h₂ 0 h₁) rfl
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      n : Nat
      l : List Nat
      h₁ : Eq l.prod n
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      ⊢ ∀ (p : Nat), Membership.mem l p → _root_.Prime p
    -/
  · simp_rw [← prime_iff]
    /-
      case refine_2
      n : Nat
      l : List Nat
      h₁ : Eq l.prod n
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      ⊢ ∀ (p : Nat), Membership.mem l p → Nat.Prime p
    -/
    exact h₂
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      n : Nat
      l : List Nat
      h₁ : Eq l.prod n
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      ⊢ ∀ (p : Nat), Membership.mem n.primeFactorsList p → _root_.Prime p
    -/
  · simp_rw [← prime_iff]
    /-
      case refine_3
      n : Nat
      l : List Nat
      h₁ : Eq l.prod n
      h₂ : ∀ (p : Nat), Membership.mem l p → Nat.Prime p
      ⊢ ∀ (p : Nat), Membership.mem n.primeFactorsList p → Nat.Prime p
    -/
    exact fun p => prime_of_mem_primeFactorsList
    /-
      🎉 no goals
    -/


theorem Prime.primeFactorsList_pow {p : ℕ} (hp : p.Prime) (n : ℕ) :
    (p ^ n).primeFactorsList = List.replicate n p := by
  /-
    p : Nat
    hp : Nat.Prime p
    n : Nat
    ⊢ Eq (HPow.hPow p n).primeFactorsList (List.replicate n p)
  -/
  symm
  /-
    p : Nat
    hp : Nat.Prime p
    n : Nat
    ⊢ Eq (List.replicate n p) (HPow.hPow p n).primeFactorsList
  -/
  rw [← List.replicate_perm]
  /-
    p : Nat
    hp : Nat.Prime p
    n : Nat
    ⊢ (List.replicate n p).Perm (HPow.hPow p n).primeFactorsList
  -/
  apply Nat.primeFactorsList_unique (List.prod_replicate n p)
  /-
    p : Nat
    hp : Nat.Prime p
    n : Nat
    ⊢ ∀ (p_1 : Nat), Membership.mem (List.replicate n p) p_1 → Nat.Prime p_1
  -/
  intro q hq
  /-
    p : Nat
    hp : Nat.Prime p
    n q : Nat
    hq : Membership.mem (List.replicate n p) q
    ⊢ Nat.Prime q
  -/
  rwa [eq_of_mem_replicate hq]
  /-
    🎉 no goals
  -/


theorem eq_prime_pow_of_unique_prime_dvd {n p : ℕ} (hpos : n ≠ 0)
    (h : ∀ {d}, Nat.Prime d → d ∣ n → d = p) : n = p ^ n.primeFactorsList.length := by
  /-
    n p : Nat
    hpos : Ne n 0
    h : ∀ {d : Nat}, Nat.Prime d → Dvd.dvd d n → Eq d p
    ⊢ Eq n (HPow.hPow p n.primeFactorsList.length)
  -/
  set k := n.primeFactorsList.length
  rw [← prod_primeFactorsList hpos, ← prod_replicate k p, eq_replicate_of_mem fun d hd =>
    h (prime_of_mem_primeFactorsList hd) (dvd_of_mem_primeFactorsList hd)]


/-- For positive `a` and `b`, the prime factors of `a * b` are the union of those of `a` and `b` -/
theorem perm_primeFactorsList_mul {a b : ℕ} (ha : a ≠ 0) (hb : b ≠ 0) :
    (a * b).primeFactorsList ~ a.primeFactorsList ++ b.primeFactorsList := by
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    ⊢ (HMul.hMul a b).primeFactorsList.Perm (HAppend.hAppend a.primeFactorsList b. …
  -/
  refine (primeFactorsList_unique ?_ ?_).symm
    /-
      case refine_1
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ Eq (HAppend.hAppend a.primeFactorsList b.primeFactorsList).prod (HMul.hMul a …
    -/
  · rw [List.prod_append, prod_primeFactorsList ha, prod_primeFactorsList hb]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      ⊢ ∀ (p : Nat), Membership.mem (HAppend.hAppend a.primeFactorsList b.primeFacto …
    -/
  · intro p hp
    /-
      case refine_2
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      p : Nat
      hp : Membership.mem (HAppend.hAppend a.primeFactorsList b.primeFactorsList) p
      ⊢ Nat.Prime p
    -/
    rw [List.mem_append] at hp
    /-
      case refine_2
      a b : Nat
      ha : Ne a 0
      hb : Ne b 0
      p : Nat
      hp : Or (Membership.mem a.primeFactorsList p) (Membership.mem b.primeFactorsLi …
      ⊢ Nat.Prime p
    -/
                               /-
                                 🎉 no goals
                               -/
    cases' hp with hp' hp' <;> exact prime_of_mem_primeFactorsList hp'
                               /-
                                 🎉 no goals
                               -/


/-- For coprime `a` and `b`, the prime factors of `a * b` are the union of those of `a` and `b` -/
theorem perm_primeFactorsList_mul_of_coprime {a b : ℕ} (hab : Coprime a b) :
    (a * b).primeFactorsList ~ a.primeFactorsList ++ b.primeFactorsList := by
  /-
    a b : Nat
    hab : a.Coprime b
    ⊢ (HMul.hMul a b).primeFactorsList.Perm (HAppend.hAppend a.primeFactorsList b. …
  -/
  rcases a.eq_zero_or_pos with (rfl | ha)
    /-
      case inl
      b : Nat
      hab : Nat.Coprime 0 b
      ⊢ (HMul.hMul 0 b).primeFactorsList.Perm (HAppend.hAppend (Nat.primeFactorsList …
    -/
  · simp [(coprime_zero_left _).mp hab]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    hab : a.Coprime b
    ha : GT.gt a 0
    ⊢ (HMul.hMul a b).primeFactorsList.Perm (HAppend.hAppend a.primeFactorsList b. …
  -/
  rcases b.eq_zero_or_pos with (rfl | hb)
    /-
      case inr.inl
      a : Nat
      ha : GT.gt a 0
      hab : a.Coprime 0
      ⊢ (HMul.hMul a 0).primeFactorsList.Perm (HAppend.hAppend a.primeFactorsList (N …
    -/
  · simp [(coprime_zero_right _).mp hab]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Nat
    hab : a.Coprime b
    ha : GT.gt a 0
    hb : GT.gt b 0
    ⊢ (HMul.hMul a b).primeFactorsList.Perm (HAppend.hAppend a.primeFactorsList b. …
  -/
  exact perm_primeFactorsList_mul ha.ne' hb.ne'
  /-
    🎉 no goals
  -/


theorem primeFactorsList_sublist_right {n k : ℕ} (h : k ≠ 0) :
    n.primeFactorsList <+ (n * k).primeFactorsList := by
  /-
    n k : Nat
    h : Ne k 0
    ⊢ n.primeFactorsList.Sublist (HMul.hMul n k).primeFactorsList
  -/
  cases' n with hn
    /-
      case zero
      k : Nat
      h : Ne k 0
      ⊢ (Nat.primeFactorsList 0).Sublist (HMul.hMul 0 k).primeFactorsList
    -/
  · simp [zero_mul]
    /-
      🎉 no goals
    -/
  /-
    case succ
    k : Nat
    h : Ne k 0
    hn : Nat
    ⊢ (HAdd.hAdd hn 1).primeFactorsList.Sublist (HMul.hMul (HAdd.hAdd hn 1) k).pri …
  -/
  apply sublist_of_subperm_of_sorted _ (primeFactorsList_sorted _) (primeFactorsList_sorted _)
  /-
    k : Nat
    h : Ne k 0
    hn : Nat
    ⊢ (HAdd.hAdd hn 1).primeFactorsList.Subperm (HMul.hMul (HAdd.hAdd hn 1) k).pri …
  -/
  simp only [(perm_primeFactorsList_mul (Nat.succ_ne_zero _) h).subperm_left]
  /-
    k : Nat
    h : Ne k 0
    hn : Nat
    ⊢ (HAdd.hAdd hn 1).primeFactorsList.Subperm (HAppend.hAppend hn.succ.primeFact …
  -/
  exact (sublist_append_left _ _).subperm
  /-
    🎉 no goals
  -/


theorem primeFactorsList_sublist_of_dvd {n k : ℕ} (h : n ∣ k) (h' : k ≠ 0) :
    n.primeFactorsList <+ k.primeFactorsList := by
  /-
    n k : Nat
    h : Dvd.dvd n k
    h' : Ne k 0
    ⊢ n.primeFactorsList.Sublist k.primeFactorsList
  -/
  obtain ⟨a, rfl⟩ := h
  /-
    case intro
    n a : Nat
    h' : Ne (HMul.hMul n a) 0
    ⊢ n.primeFactorsList.Sublist (HMul.hMul n a).primeFactorsList
  -/
  exact primeFactorsList_sublist_right (right_ne_zero_of_mul h')
  /-
    🎉 no goals
  -/


theorem primeFactorsList_subset_right {n k : ℕ} (h : k ≠ 0) :
    n.primeFactorsList ⊆ (n * k).primeFactorsList :=
  (primeFactorsList_sublist_right h).subset


theorem primeFactorsList_subset_of_dvd {n k : ℕ} (h : n ∣ k) (h' : k ≠ 0) :
    n.primeFactorsList ⊆ k.primeFactorsList :=
  (primeFactorsList_sublist_of_dvd h h').subset


theorem dvd_of_primeFactorsList_subperm {a b : ℕ} (ha : a ≠ 0)
    (h : a.primeFactorsList <+~ b.primeFactorsList) : a ∣ b := by
  /-
    a b : Nat
    ha : Ne a 0
    h : a.primeFactorsList.Subperm b.primeFactorsList
    ⊢ Dvd.dvd a b
  -/
  rcases b.eq_zero_or_pos with (rfl | hb)
    /-
      case inl
      a : Nat
      ha : Ne a 0
      h : a.primeFactorsList.Subperm (Nat.primeFactorsList 0)
      ⊢ Dvd.dvd a 0
    -/
  · exact dvd_zero _
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    ha : Ne a 0
    h : a.primeFactorsList.Subperm b.primeFactorsList
    hb : GT.gt b 0
    ⊢ Dvd.dvd a b
  -/
  rcases a with (_ | _ | a)
    /-
      case inr.zero
      b : Nat
      hb : GT.gt b 0
      ha : Ne 0 0
      h : (Nat.primeFactorsList 0).Subperm b.primeFactorsList
      ⊢ Dvd.dvd 0 b
    -/
  · exact (ha rfl).elim
    /-
      🎉 no goals
    -/
    /-
      case inr.succ.zero
      b : Nat
      hb : GT.gt b 0
      ha : Ne (HAdd.hAdd 0 1) 0
      h : (HAdd.hAdd 0 1).primeFactorsList.Subperm b.primeFactorsList
      ⊢ Dvd.dvd (HAdd.hAdd 0 1) b
    -/
  · exact one_dvd _
    /-
      🎉 no goals
    -/
  -- Porting note: previous proof
  --use (b.primeFactorsList.diff a.succ.succ.primeFactorsList).prod
  /-
    case inr.succ.succ
    b : Nat
    hb : GT.gt b 0
    a : Nat
    ha : Ne (HAdd.hAdd (HAdd.hAdd a 1) 1) 0
    h : (HAdd.hAdd (HAdd.hAdd a 1) 1).primeFactorsList.Subperm b.primeFactorsList
    ⊢ Dvd.dvd (HAdd.hAdd (HAdd.hAdd a 1) 1) b
  -/
  use (@List.diff _ instBEqOfDecidableEq b.primeFactorsList a.succ.succ.primeFactorsList).prod
  /-
    case h
    b : Nat
    hb : GT.gt b 0
    a : Nat
    ha : Ne (HAdd.hAdd (HAdd.hAdd a 1) 1) 0
    h : (HAdd.hAdd (HAdd.hAdd a 1) 1).primeFactorsList.Subperm b.primeFactorsList
    ⊢ Eq b (HMul.hMul (HAdd.hAdd (HAdd.hAdd a 1) 1) (b.primeFactorsList.diff a.suc …
  -/
  nth_rw 1 [← Nat.prod_primeFactorsList ha]
  rw [← List.prod_append,
    List.Perm.prod_eq <| List.subperm_append_diff_self_of_count_le <| List.subperm_ext_iff.mp h,
    Nat.prod_primeFactorsList hb.ne']


theorem replicate_subperm_primeFactorsList_iff {a b n : ℕ} (ha : Prime a) (hb : b ≠ 0) :
    replicate n a <+~ primeFactorsList b ↔ a ^ n ∣ b := by
  induction n generalizing b with
  | zero => simp
  | succ n ih =>
    constructor
    · rw [List.subperm_iff]
      rintro ⟨u, hu1, hu2⟩
      rw [← Nat.prod_primeFactorsList hb, ← hu1.prod_eq, ← prod_replicate]
      exact hu2.prod_dvd_prod
    · rintro ⟨c, rfl⟩
      rw [Ne, pow_succ', mul_assoc, mul_eq_zero, _root_.not_or] at hb
      rw [pow_succ', mul_assoc, replicate_succ,
        (Nat.perm_primeFactorsList_mul hb.1 hb.2).subperm_left, primeFactorsList_prime ha,
        singleton_append, subperm_cons, ih hb.2]
      exact dvd_mul_right _ _


theorem mem_primeFactorsList_mul {a b : ℕ} (ha : a ≠ 0) (hb : b ≠ 0) {p : ℕ} :
    p ∈ (a * b).primeFactorsList ↔ p ∈ a.primeFactorsList ∨ p ∈ b.primeFactorsList := by
  rw [mem_primeFactorsList (mul_ne_zero ha hb), mem_primeFactorsList ha, mem_primeFactorsList hb,
    ← and_or_left]
  /-
    a b : Nat
    ha : Ne a 0
    hb : Ne b 0
    p : Nat
    ⊢ Iff (And (Nat.Prime p) (Dvd.dvd p (HMul.hMul a b))) (And (Nat.Prime p) (Or ( …
  -/
  simpa only [and_congr_right_iff] using Prime.dvd_mul
  /-
    🎉 no goals
  -/


/-- The sets of factors of coprime `a` and `b` are disjoint -/
theorem coprime_primeFactorsList_disjoint {a b : ℕ} (hab : a.Coprime b) :
    List.Disjoint a.primeFactorsList b.primeFactorsList := by
  /-
    a b : Nat
    hab : a.Coprime b
    ⊢ a.primeFactorsList.Disjoint b.primeFactorsList
  -/
  intro q hqa hqb
  /-
    a b : Nat
    hab : a.Coprime b
    q : Nat
    hqa : Membership.mem a.primeFactorsList q
    hqb : Membership.mem b.primeFactorsList q
    ⊢ False
  -/
  apply not_prime_one
  rw [← eq_one_of_dvd_coprimes hab (dvd_of_mem_primeFactorsList hqa)
    (dvd_of_mem_primeFactorsList hqb)]
  /-
    a b : Nat
    hab : a.Coprime b
    q : Nat
    hqa : Membership.mem a.primeFactorsList q
    hqb : Membership.mem b.primeFactorsList q
    ⊢ Nat.Prime q
  -/
  exact prime_of_mem_primeFactorsList hqa
  /-
    🎉 no goals
  -/


theorem mem_primeFactorsList_mul_of_coprime {a b : ℕ} (hab : Coprime a b) (p : ℕ) :
    p ∈ (a * b).primeFactorsList ↔ p ∈ a.primeFactorsList ∪ b.primeFactorsList := by
  /-
    a b : Nat
    hab : a.Coprime b
    p : Nat
    ⊢ Iff (Membership.mem (HMul.hMul a b).primeFactorsList p) (Membership.mem (Uni …
  -/
  rcases a.eq_zero_or_pos with (rfl | ha)
    /-
      case inl
      b p : Nat
      hab : Nat.Coprime 0 b
      ⊢ Iff (Membership.mem (HMul.hMul 0 b).primeFactorsList p) (Membership.mem (Uni …
    -/
  · simp [(coprime_zero_left _).mp hab]
    /-
      🎉 no goals
    -/
  /-
    case inr
    a b : Nat
    hab : a.Coprime b
    p : Nat
    ha : GT.gt a 0
    ⊢ Iff (Membership.mem (HMul.hMul a b).primeFactorsList p) (Membership.mem (Uni …
  -/
  rcases b.eq_zero_or_pos with (rfl | hb)
    /-
      case inr.inl
      a p : Nat
      ha : GT.gt a 0
      hab : a.Coprime 0
      ⊢ Iff (Membership.mem (HMul.hMul a 0).primeFactorsList p) (Membership.mem (Uni …
    -/
  · simp [(coprime_zero_right _).mp hab]
    /-
      🎉 no goals
    -/
  /-
    case inr.inr
    a b : Nat
    hab : a.Coprime b
    p : Nat
    ha : GT.gt a 0
    hb : GT.gt b 0
    ⊢ Iff (Membership.mem (HMul.hMul a b).primeFactorsList p) (Membership.mem (Uni …
  -/
  rw [mem_primeFactorsList_mul ha.ne' hb.ne', List.mem_union_iff]
  /-
    🎉 no goals
  -/


/-- If `p` is a prime factor of `a` then `p` is also a prime factor of `a * b` for any `b > 0` -/
theorem mem_primeFactorsList_mul_left {p a b : ℕ} (hpa : p ∈ a.primeFactorsList) (hb : b ≠ 0) :
    p ∈ (a * b).primeFactorsList := by
  /-
    p a b : Nat
    hpa : Membership.mem a.primeFactorsList p
    hb : Ne b 0
    ⊢ Membership.mem (HMul.hMul a b).primeFactorsList p
  -/
  rcases eq_or_ne a 0 with (rfl | ha)
    /-
      case inl
      p b : Nat
      hb : Ne b 0
      hpa : Membership.mem (Nat.primeFactorsList 0) p
      ⊢ Membership.mem (HMul.hMul 0 b).primeFactorsList p
    -/
  · simp at hpa
    /-
      🎉 no goals
    -/
  /-
    case inr
    p a b : Nat
    hpa : Membership.mem a.primeFactorsList p
    hb : Ne b 0
    ha : Ne a 0
    ⊢ Membership.mem (HMul.hMul a b).primeFactorsList p
  -/
  apply (mem_primeFactorsList_mul ha hb).2 (Or.inl hpa)
  /-
    🎉 no goals
  -/


/-- If `p` is a prime factor of `b` then `p` is also a prime factor of `a * b` for any `a > 0` -/
theorem mem_primeFactorsList_mul_right {p a b : ℕ} (hpb : p ∈ b.primeFactorsList) (ha : a ≠ 0) :
    p ∈ (a * b).primeFactorsList := by
  /-
    p a b : Nat
    hpb : Membership.mem b.primeFactorsList p
    ha : Ne a 0
    ⊢ Membership.mem (HMul.hMul a b).primeFactorsList p
  -/
  rw [mul_comm]
  /-
    p a b : Nat
    hpb : Membership.mem b.primeFactorsList p
    ha : Ne a 0
    ⊢ Membership.mem (HMul.hMul b a).primeFactorsList p
  -/
  exact mem_primeFactorsList_mul_left hpb ha
  /-
    🎉 no goals
  -/


theorem eq_two_pow_or_exists_odd_prime_and_dvd (n : ℕ) :
    (∃ k : ℕ, n = 2 ^ k) ∨ ∃ p, Nat.Prime p ∧ p ∣ n ∧ Odd p :=
  (eq_or_ne n 0).elim (fun hn => Or.inr ⟨3, prime_three, hn.symm ▸ dvd_zero 3, ⟨1, rfl⟩⟩) fun hn =>
    or_iff_not_imp_right.mpr fun H =>
      ⟨n.primeFactorsList.length,
        eq_prime_pow_of_unique_prime_dvd hn fun {_} hprime hdvd =>
          hprime.eq_two_or_odd'.resolve_right fun hodd => H ⟨_, hprime, hdvd, hodd⟩⟩


theorem four_dvd_or_exists_odd_prime_and_dvd_of_two_lt {n : ℕ} (n2 : 2 < n) :
    4 ∣ n ∨ ∃ p, Prime p ∧ p ∣ n ∧ Odd p := by
  /-
    n : Nat
    n2 : LT.lt 2 n
    ⊢ Or (Dvd.dvd 4 n) (Exists fun p => And (Nat.Prime p) (And (Dvd.dvd p n) (Odd  …
  -/
  obtain ⟨_ | _ | k, rfl⟩ | ⟨p, hp, hdvd, hodd⟩ := n.eq_two_pow_or_exists_odd_prime_and_dvd
    /-
      case inl.intro.zero
      n2 : LT.lt 2 (HPow.hPow 2 0)
      ⊢ Or (Dvd.dvd 4 (HPow.hPow 2 0)) (Exists fun p => And (Nat.Prime p) (And (Dvd. …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case inl.intro.succ.zero
      n2 : LT.lt 2 (HPow.hPow 2 (HAdd.hAdd 0 1))
      ⊢ Or (Dvd.dvd 4 (HPow.hPow 2 (HAdd.hAdd 0 1))) (Exists fun p => And (Nat.Prime …
    -/
  · contradiction
    /-
      🎉 no goals
    -/
    /-
      case inl.intro.succ.succ
      k : Nat
      n2 : LT.lt 2 (HPow.hPow 2 (HAdd.hAdd (HAdd.hAdd k 1) 1))
      ⊢ Or (Dvd.dvd 4 (HPow.hPow 2 (HAdd.hAdd (HAdd.hAdd k 1) 1))) (Exists fun p =>  …
    -/
  · simp [Nat.pow_succ, mul_assoc]
    /-
      🎉 no goals
    -/
    /-
      case inr.intro.intro.intro
      n : Nat
      n2 : LT.lt 2 n
      p : Nat
      hp : Nat.Prime p
      hdvd : Dvd.dvd p n
      hodd : Odd p
      ⊢ Or (Dvd.dvd 4 n) (Exists fun p => And (Nat.Prime p) (And (Dvd.dvd p n) (Odd  …
    -/
  · exact Or.inr ⟨p, hp, hdvd, hodd⟩
    /-
      🎉 no goals
    -/


