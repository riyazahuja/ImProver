@[simp]
theorem coeff_add (p q : R[X]) (n : ℕ) : coeff (p + q) n = coeff p n + coeff q n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    ⊢ Eq ((HAdd.hAdd p q).coeff n) (HAdd.hAdd (p.coeff n) (q.coeff n))
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    q : Polynomial R
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((HAdd.hAdd { toFinsupp := toFinsupp✝ } q).coeff n) (HAdd.hAdd ({ toFinsu …
  -/
  rcases q with ⟨⟩
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    toFinsupp✝¹ toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((HAdd.hAdd { toFinsupp := toFinsupp✝¹ } { toFinsupp := toFinsupp✝ }).coe …
  -/
  simp_rw [← ofFinsupp_add, coeff]
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    toFinsupp✝¹ toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((HAdd.hAdd toFinsupp✝¹ toFinsupp✝) n) (HAdd.hAdd (toFinsupp✝¹ n) (toFins …
  -/
  exact Finsupp.add_apply _ _ _
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_smul [SMulZeroClass S R] (r : S) (p : R[X]) (n : ℕ) :
    coeff (r • p) n = r • coeff p n := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : SMulZeroClass S R
    r : S
    p : Polynomial R
    n : Nat
    ⊢ Eq ((HSMul.hSMul r p).coeff n) (HSMul.hSMul r (p.coeff n))
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : SMulZeroClass S R
    r : S
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((HSMul.hSMul r { toFinsupp := toFinsupp✝ }).coeff n) (HSMul.hSMul r ({ t …
  -/
  simp_rw [← ofFinsupp_smul, coeff]
  /-
    case ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : SMulZeroClass S R
    r : S
    n : Nat
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq ((HSMul.hSMul r toFinsupp✝) n) (HSMul.hSMul r (toFinsupp✝ n))
  -/
  exact Finsupp.smul_apply _ _ _
  /-
    🎉 no goals
  -/


theorem support_smul [SMulZeroClass S R] (r : S) (p : R[X]) :
    support (r • p) ⊆ support p := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : SMulZeroClass S R
    r : S
    p : Polynomial R
    ⊢ HasSubset.Subset (HSMul.hSMul r p).support p.support
  -/
  intro i hi
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : SMulZeroClass S R
    r : S
    p : Polynomial R
    i : Nat
    hi : Membership.mem (HSMul.hSMul r p).support i
    ⊢ Membership.mem p.support i
  -/
  simp? [mem_support_iff] at hi ⊢ says simp only [mem_support_iff, coeff_smul, ne_eq] at hi ⊢
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : SMulZeroClass S R
    r : S
    p : Polynomial R
    i : Nat
    hi : Not (Eq (HSMul.hSMul r (p.coeff i)) 0)
    ⊢ Not (Eq (p.coeff i) 0)
  -/
  contrapose! hi
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : SMulZeroClass S R
    r : S
    p : Polynomial R
    i : Nat
    hi : Eq (p.coeff i) 0
    ⊢ Eq (HSMul.hSMul r (p.coeff i)) 0
  -/
  simp [hi]
  /-
    🎉 no goals
  -/


open scoped Pointwise in
theorem card_support_mul_le : #(p * q).support ≤ #p.support * #q.support := by
  calc #(p * q).support
   _ = #(p.toFinsupp * q.toFinsupp).support := by rw [← support_toFinsupp, toFinsupp_mul]
   _ ≤ #(p.toFinsupp.support + q.toFinsupp.support) :=
    Finset.card_le_card (AddMonoidAlgebra.support_mul p.toFinsupp q.toFinsupp)
   _ ≤ #p.support * #q.support := Finset.card_image₂_le ..


/-- `Polynomial.sum` as a linear map. -/
@[simps]
def lsum {R A M : Type*} [Semiring R] [Semiring A] [AddCommMonoid M] [Module R A] [Module R M]
    (f : ℕ → A →ₗ[R] M) : A[X] →ₗ[R] M where
  toFun p := p.sum (f · ·)
  map_add' p q := sum_add_index p q _ (fun n => (f n).map_zero) fun n _ _ => (f n).map_add _ _
  map_smul' c p := by
    -- Porting note: added `dsimp only`; `beta_reduce` alone is not sufficient
    /-
      R✝ : Type u
      S : Type v
      a b : R✝
      n m : Nat
      inst✝⁵ : Semiring R✝
      p✝ q r : Polynomial R✝
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R A
      inst✝ : Module R M
      f : Nat → LinearMap (RingHom.id R) A M
      c : R
      p : Polynomial A
      ⊢ Eq ({ toFun := fun p => p.sum fun x1 x2 => (f x1) x2, map_add' := ⋯ }.toFun  …
    -/
    dsimp only
    /-
      R✝ : Type u
      S : Type v
      a b : R✝
      n m : Nat
      inst✝⁵ : Semiring R✝
      p✝ q r : Polynomial R✝
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R A
      inst✝ : Module R M
      f : Nat → LinearMap (RingHom.id R) A M
      c : R
      p : Polynomial A
      ⊢ Eq ((HSMul.hSMul c p).sum fun x1 x2 => (f x1) x2) (HSMul.hSMul ((RingHom.id  …
    -/
    rw [sum_eq_of_subset (f · ·) (fun n => (f n).map_zero) (support_smul c p)]
    /-
      R✝ : Type u
      S : Type v
      a b : R✝
      n m : Nat
      inst✝⁵ : Semiring R✝
      p✝ q r : Polynomial R✝
      R : Type u_1
      A : Type u_2
      M : Type u_3
      inst✝⁴ : Semiring R
      inst✝³ : Semiring A
      inst✝² : AddCommMonoid M
      inst✝¹ : Module R A
      inst✝ : Module R M
      f : Nat → LinearMap (RingHom.id R) A M
      c : R
      p : Polynomial A
      ⊢ Eq (p.support.sum fun n => (f n) ((HSMul.hSMul c p).coeff n)) (HSMul.hSMul ( …
    -/
    simp only [sum_def, Finset.smul_sum, coeff_smul, LinearMap.map_smul, RingHom.id_apply]
    /-
      🎉 no goals
    -/


/-- The nth coefficient, as a linear map. -/
def lcoeff (n : ℕ) : R[X] →ₗ[R] R where
  toFun p := coeff p n
  map_add' p q := coeff_add p q n
  map_smul' r p := coeff_smul r p n


@[simp]
theorem lcoeff_apply (n : ℕ) (f : R[X]) : lcoeff R n f = coeff f n :=
  rfl


@[simp]
theorem finset_sum_coeff {ι : Type*} (s : Finset ι) (f : ι → R[X]) (n : ℕ) :
    coeff (∑ b ∈ s, f b) n = ∑ b ∈ s, coeff (f b) n :=
  map_sum (lcoeff R n) _ _


lemma coeff_list_sum (l : List R[X]) (n : ℕ) :
    l.sum.coeff n = (l.map (lcoeff R n)).sum :=
  map_list_sum (lcoeff R n) _


lemma coeff_list_sum_map {ι : Type*} (l : List ι) (f : ι → R[X]) (n : ℕ) :
    (l.map f).sum.coeff n = (l.map (fun a => (f a).coeff n)).sum := by
  /-
    R : Type u
    inst✝ : Semiring R
    ι : Type u_1
    l : List ι
    f : ι → Polynomial R
    n : Nat
    ⊢ Eq ((List.map f l).sum.coeff n) (List.map (fun a => (f a).coeff n) l).sum
  -/
  simp_rw [coeff_list_sum, List.map_map, Function.comp_def, lcoeff_apply]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_sum [Semiring S] (n : ℕ) (f : ℕ → R → S[X]) :
    coeff (p.sum f) n = p.sum fun a b => coeff (f a b) n := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p : Polynomial R
    inst✝ : Semiring S
    n : Nat
    f : Nat → R → Polynomial S
    ⊢ Eq ((p.sum f).coeff n) (p.sum fun a b => (f a b).coeff n)
  -/
  rcases p with ⟨⟩
  /-
    case ofFinsupp
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    n : Nat
    f : Nat → R → Polynomial S
    toFinsupp✝ : AddMonoidAlgebra R Nat
    ⊢ Eq (({ toFinsupp := toFinsupp✝ }.sum f).coeff n) ({ toFinsupp := toFinsupp✝  …
  -/
  simp [Polynomial.sum, support_ofFinsupp, coeff_ofFinsupp]
  /-
    🎉 no goals
  -/


/-- Decomposes the coefficient of the product `p * q` as a sum
over `antidiagonal`. A version which sums over `range (n + 1)` can be obtained
by using `Finset.Nat.sum_antidiagonal_eq_sum_range_succ`. -/
theorem coeff_mul (p q : R[X]) (n : ℕ) :
    coeff (p * q) n = ∑ x ∈ antidiagonal n, coeff p x.1 * coeff q x.2 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    n : Nat
    ⊢ Eq ((HMul.hMul p q).coeff n) ((Finset.HasAntidiagonal.antidiagonal n).sum fu …
  -/
  rcases p with ⟨p⟩; rcases q with ⟨q⟩
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p q : AddMonoidAlgebra R Nat
    ⊢ Eq ((HMul.hMul { toFinsupp := p } { toFinsupp := q }).coeff n) ((Finset.HasA …
  -/
  simp_rw [← ofFinsupp_mul, coeff]
  /-
    case ofFinsupp.ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    p q : AddMonoidAlgebra R Nat
    ⊢ Eq ((HMul.hMul p q) n) ((Finset.HasAntidiagonal.antidiagonal n).sum fun x => …
  -/
  exact AddMonoidAlgebra.mul_apply_antidiagonal p q n _ Finset.mem_antidiagonal
  /-
    🎉 no goals
  -/


@[simp]
                                                                                    /-
                                                                                      R : Type u
                                                                                      inst✝ : Semiring R
                                                                                      p q : Polynomial R
                                                                                      ⊢ Eq ((HMul.hMul p q).coeff 0) (HMul.hMul (p.coeff 0) (q.coeff 0))
                                                                                    -/
theorem mul_coeff_zero (p q : R[X]) : coeff (p * q) 0 = coeff p 0 * coeff q 0 := by simp [coeff_mul]
                                                                                    /-
                                                                                      🎉 no goals
                                                                                    -/


theorem mul_coeff_one (p q : R[X]) :
    coeff (p * q) 1 = coeff p 0 * coeff q 1 + coeff p 1 * coeff q 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ Eq ((HMul.hMul p q).coeff 1) (HAdd.hAdd (HMul.hMul (p.coeff 0) (q.coeff 1))  …
  -/
  rw [coeff_mul, Nat.antidiagonal_eq_map]
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    ⊢ Eq ((Finset.map { toFun := fun i => { fst := i, snd := HSub.hSub 1 i }, inj' …
  -/
  simp [sum_range_succ]
  /-
    🎉 no goals
  -/


/-- `constantCoeff p` returns the constant term of the polynomial `p`,
  defined as `coeff p 0`. This is a ring homomorphism. -/
@[simps]
def constantCoeff : R[X] →+* R where
  toFun p := coeff p 0
  map_one' := coeff_one_zero
  map_mul' := mul_coeff_zero
  map_zero' := coeff_zero 0
  map_add' p q := coeff_add p q 0


theorem isUnit_C {x : R} : IsUnit (C x) ↔ IsUnit x :=
  ⟨fun h => (congr_arg IsUnit coeff_C_zero).mp (h.map <| @constantCoeff R _), fun h => h.map C⟩


                                                                /-
                                                                  R : Type u
                                                                  inst✝ : Semiring R
                                                                  p : Polynomial R
                                                                  ⊢ Eq ((HMul.hMul p Polynomial.X).coeff 0) 0
                                                                -/
theorem coeff_mul_X_zero (p : R[X]) : coeff (p * X) 0 = 0 := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


                                                                /-
                                                                  R : Type u
                                                                  inst✝ : Semiring R
                                                                  p : Polynomial R
                                                                  ⊢ Eq ((HMul.hMul Polynomial.X p).coeff 0) 0
                                                                -/
theorem coeff_X_mul_zero (p : R[X]) : coeff (X * p) 0 = 0 := by simp
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem coeff_C_mul_X_pow (x : R) (k n : ℕ) :
    coeff (C x * X ^ k : R[X]) n = if n = k then x else 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    x : R
    k n : Nat
    ⊢ Eq ((HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k)).coeff n) (ite (E …
  -/
  rw [C_mul_X_pow_eq_monomial, coeff_monomial]
  /-
    R : Type u
    inst✝ : Semiring R
    x : R
    k n : Nat
    ⊢ Eq (ite (Eq k n) x 0) (ite (Eq n k) x 0)
  -/
  congr 1
  /-
    case e_c
    R : Type u
    inst✝ : Semiring R
    x : R
    k n : Nat
    ⊢ Eq (Eq k n) (Eq n k)
  -/
  simp [eq_comm]
  /-
    🎉 no goals
  -/


theorem coeff_C_mul_X (x : R) (n : ℕ) : coeff (C x * X : R[X]) n = if n = 1 then x else 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    x : R
    n : Nat
    ⊢ Eq ((HMul.hMul (Polynomial.C x) Polynomial.X).coeff n) (ite (Eq n 1) x 0)
  -/
  rw [← pow_one X, coeff_C_mul_X_pow]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_C_mul (p : R[X]) : coeff (C a * p) n = a * coeff p n := by
  /-
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    p : Polynomial R
    ⊢ Eq ((HMul.hMul (Polynomial.C a) p).coeff n) (HMul.hMul a (p.coeff n))
  -/
  rcases p with ⟨p⟩
  /-
    case ofFinsupp
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    p : AddMonoidAlgebra R Nat
    ⊢ Eq ((HMul.hMul (Polynomial.C a) { toFinsupp := p }).coeff n) (HMul.hMul a ({ …
  -/
  simp_rw [← monomial_zero_left, ← ofFinsupp_single, ← ofFinsupp_mul, coeff]
  /-
    case ofFinsupp
    R : Type u
    a : R
    n : Nat
    inst✝ : Semiring R
    p : AddMonoidAlgebra R Nat
    ⊢ Eq ((HMul.hMul (Finsupp.single 0 a) p) n) (HMul.hMul a (p n))
  -/
  exact AddMonoidAlgebra.single_zero_mul_apply p a n
  /-
    🎉 no goals
  -/


theorem C_mul' (a : R) (f : R[X]) : C a * f = a • f := by
  /-
    R : Type u
    inst✝ : Semiring R
    a : R
    f : Polynomial R
    ⊢ Eq (HMul.hMul (Polynomial.C a) f) (HSMul.hSMul a f)
  -/
  ext
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    a : R
    f : Polynomial R
    n✝ : Nat
    ⊢ Eq ((HMul.hMul (Polynomial.C a) f).coeff n✝) ((HSMul.hSMul a f).coeff n✝)
  -/
  rw [coeff_C_mul, coeff_smul, smul_eq_mul]
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_mul_C (p : R[X]) (n : ℕ) (a : R) : coeff (p * C a) n = coeff p n * a := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    a : R
    ⊢ Eq ((HMul.hMul p (Polynomial.C a)).coeff n) (HMul.hMul (p.coeff n) a)
  -/
  rcases p with ⟨p⟩
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    p : AddMonoidAlgebra R Nat
    ⊢ Eq ((HMul.hMul { toFinsupp := p } (Polynomial.C a)).coeff n) (HMul.hMul ({ t …
  -/
  simp_rw [← monomial_zero_left, ← ofFinsupp_single, ← ofFinsupp_mul, coeff]
  /-
    case ofFinsupp
    R : Type u
    inst✝ : Semiring R
    n : Nat
    a : R
    p : AddMonoidAlgebra R Nat
    ⊢ Eq ((HMul.hMul p (Finsupp.single 0 a)) n) (HMul.hMul (p n) a)
  -/
  exact AddMonoidAlgebra.mul_single_zero_apply p a n
  /-
    🎉 no goals
  -/


@[simp] lemma coeff_mul_natCast {a k : ℕ} :
  coeff (p * (a : R[X])) k = coeff p k * (↑a : R) := coeff_mul_C _ _ _


@[simp] lemma coeff_natCast_mul {a k : ℕ} :
  coeff ((a : R[X]) * p) k = a * coeff p k := coeff_C_mul _


@[simp] lemma coeff_mul_ofNat {a k : ℕ} [Nat.AtLeastTwo a] :
  coeff (p * (ofNat(a) : R[X])) k = coeff p k * ofNat(a) := coeff_mul_C _ _ _


@[simp] lemma coeff_ofNat_mul {a k : ℕ} [Nat.AtLeastTwo a] :
  coeff ((ofNat(a) : R[X]) * p) k = ofNat(a) * coeff p k := coeff_C_mul _


@[simp] lemma coeff_mul_intCast [Ring S] {p : S[X]} {a : ℤ} {k : ℕ} :
  coeff (p * (a : S[X])) k = coeff p k * (↑a : S) := coeff_mul_C _ _ _


@[simp] lemma coeff_intCast_mul [Ring S] {p : S[X]} {a : ℤ} {k : ℕ} :
  coeff ((a : S[X]) * p) k = a * coeff p k := coeff_C_mul _


@[simp]
theorem coeff_X_pow (k n : ℕ) : coeff (X ^ k : R[X]) n = if n = k then 1 else 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    k n : Nat
    ⊢ Eq ((HPow.hPow Polynomial.X k).coeff n) (ite (Eq n k) 1 0)
  -/
  simp only [one_mul, RingHom.map_one, ← coeff_C_mul_X_pow]
  /-
    🎉 no goals
  -/


                                                                    /-
                                                                      R : Type u
                                                                      inst✝ : Semiring R
                                                                      n : Nat
                                                                      ⊢ Eq ((HPow.hPow Polynomial.X n).coeff n) 1
                                                                    -/
theorem coeff_X_pow_self (n : ℕ) : coeff (X ^ n : R[X]) n = 1 := by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


theorem support_binomial {k m : ℕ} (hkm : k ≠ m) {x y : R} (hx : x ≠ 0) (hy : y ≠ 0) :
    support (C x * X ^ k + C y * X ^ m) = {k, m} := by
  /-
    R : Type u
    inst✝ : Semiring R
    k m : Nat
    hkm : Ne k m
    x y : R
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k)) (HMul. …
  -/
  apply subset_antisymm (support_binomial' k m x y)
  simp_rw [insert_subset_iff, singleton_subset_iff, mem_support_iff, coeff_add, coeff_C_mul,
    coeff_X_pow_self, mul_one, coeff_X_pow, if_neg hkm, if_neg hkm.symm, mul_zero, zero_add,
    add_zero, Ne, hx, hy, not_false_eq_true, and_true]


theorem support_trinomial {k m n : ℕ} (hkm : k < m) (hmn : m < n) {x y z : R} (hx : x ≠ 0)
    (hy : y ≠ 0) (hz : z ≠ 0) :
    support (C x * X ^ k + C y * X ^ m + C z * X ^ n) = {k, m, n} := by
  /-
    R : Type u
    inst✝ : Semiring R
    k m n : Nat
    hkm : LT.lt k m
    hmn : LT.lt m n
    x y z : R
    hx : Ne x 0
    hy : Ne y 0
    hz : Ne z 0
    ⊢ Eq (HAdd.hAdd (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X …
  -/
  apply subset_antisymm (support_trinomial' k m n x y z)
  simp_rw [insert_subset_iff, singleton_subset_iff, mem_support_iff, coeff_add, coeff_C_mul,
    coeff_X_pow_self, mul_one, coeff_X_pow, if_neg hkm.ne, if_neg hkm.ne', if_neg hmn.ne,
    if_neg hmn.ne', if_neg (hkm.trans hmn).ne, if_neg (hkm.trans hmn).ne', mul_zero, add_zero,
    zero_add, Ne, hx, hy, hz, not_false_eq_true, and_true]


theorem card_support_binomial {k m : ℕ} (h : k ≠ m) {x y : R} (hx : x ≠ 0) (hy : y ≠ 0) :
    #(support (C x * X ^ k + C y * X ^ m)) = 2 := by
  /-
    R : Type u
    inst✝ : Semiring R
    k m : Nat
    h : Ne k m
    x y : R
    hx : Ne x 0
    hy : Ne y 0
    ⊢ Eq (HAdd.hAdd (HMul.hMul (Polynomial.C x) (HPow.hPow Polynomial.X k)) (HMul. …
  -/
  rw [support_binomial h hx hy, card_insert_of_not_mem (mt mem_singleton.mp h), card_singleton]
  /-
    🎉 no goals
  -/


theorem card_support_trinomial {k m n : ℕ} (hkm : k < m) (hmn : m < n) {x y z : R} (hx : x ≠ 0)
    (hy : y ≠ 0) (hz : z ≠ 0) : #(support (C x * X ^ k + C y * X ^ m + C z * X ^ n)) = 3 := by
  rw [support_trinomial hkm hmn hx hy hz,
    card_insert_of_not_mem
      (mt mem_insert.mp (not_or_intro hkm.ne (mt mem_singleton.mp (hkm.trans hmn).ne))),
    card_insert_of_not_mem (mt mem_singleton.mp hmn.ne), card_singleton]


@[simp]
theorem coeff_mul_X_pow (p : R[X]) (n d : ℕ) :
    coeff (p * Polynomial.X ^ n) (d + n) = coeff p d := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n d : Nat
    ⊢ Eq ((HMul.hMul p (HPow.hPow Polynomial.X n)).coeff (HAdd.hAdd d n)) (p.coeff …
  -/
  rw [coeff_mul, Finset.sum_eq_single (d, n), coeff_X_pow, if_pos rfl, mul_one]
    /-
      case h₀
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n d : Nat
      ⊢ ∀ (b : Prod Nat Nat), Membership.mem (Finset.HasAntidiagonal.antidiagonal (H …
    -/
  · rintro ⟨i, j⟩ h1 h2
    /-
      case h₀.mk
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) { fs …
      h2 : Ne { fst := i, snd := j } { fst := d, snd := n }
      ⊢ Eq (HMul.hMul (p.coeff { fst := i, snd := j }.1) ((HPow.hPow Polynomial.X n) …
    -/
    rw [coeff_X_pow, if_neg, mul_zero]
    /-
      case h₀.mk.hnc
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) { fs …
      h2 : Ne { fst := i, snd := j } { fst := d, snd := n }
      ⊢ Not (Eq { fst := i, snd := j }.2 n)
    -/
    rintro rfl
    /-
      case h₀.mk.hnc
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d { fst := …
      h2 : Ne { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
      ⊢ False
    -/
    apply h2
    /-
      case h₀.mk.hnc
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      d i j : Nat
      h1 : Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d { fst := …
      h2 : Ne { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
      ⊢ Eq { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
    -/
    rw [mem_antidiagonal, add_right_cancel_iff] at h1
    /-
      case h₀.mk.hnc
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      d i j : Nat
      h1 : Eq { fst := i, snd := j }.1 d
      h2 : Ne { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
      ⊢ Eq { fst := i, snd := j } { fst := d, snd := { fst := i, snd := j }.2 }
    -/
    subst h1
    /-
      case h₀.mk.hnc
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      i j : Nat
      h2 : Ne { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := { fst …
      ⊢ Eq { fst := i, snd := j } { fst := { fst := i, snd := j }.1, snd := { fst := …
    -/
    rfl
    /-
      🎉 no goals
    -/
    /-
      case h₁
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n d : Nat
      ⊢ Not (Membership.mem (Finset.HasAntidiagonal.antidiagonal (HAdd.hAdd d n)) {  …
    -/
  · exact fun h1 => (h1 (mem_antidiagonal.2 rfl)).elim
    /-
      🎉 no goals
    -/


@[simp]
theorem coeff_X_pow_mul (p : R[X]) (n d : ℕ) :
    coeff (Polynomial.X ^ n * p) (d + n) = coeff p d := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n d : Nat
    ⊢ Eq ((HMul.hMul (HPow.hPow Polynomial.X n) p).coeff (HAdd.hAdd d n)) (p.coeff …
  -/
  rw [(commute_X_pow p n).eq, coeff_mul_X_pow]
  /-
    🎉 no goals
  -/


theorem coeff_mul_X_pow' (p : R[X]) (n d : ℕ) :
    (p * X ^ n).coeff d = ite (n ≤ d) (p.coeff (d - n)) 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n d : Nat
    ⊢ Eq ((HMul.hMul p (HPow.hPow Polynomial.X n)).coeff d) (ite (LE.le n d) (p.co …
  -/
  split_ifs with h
    /-
      case pos
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n d : Nat
      h : LE.le n d
      ⊢ Eq ((HMul.hMul p (HPow.hPow Polynomial.X n)).coeff d) (p.coeff (HSub.hSub d  …
    -/
  · rw [← tsub_add_cancel_of_le h, coeff_mul_X_pow, add_tsub_cancel_right]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n d : Nat
      h : Not (LE.le n d)
      ⊢ Eq ((HMul.hMul p (HPow.hPow Polynomial.X n)).coeff d) 0
    -/
  · refine (coeff_mul _ _ _).trans (Finset.sum_eq_zero fun x hx => ?_)
    /-
      case neg
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n d : Nat
      h : Not (LE.le n d)
      x : Prod Nat Nat
      hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) x
      ⊢ Eq (HMul.hMul (p.coeff x.1) ((HPow.hPow Polynomial.X n).coeff x.2)) 0
    -/
    rw [coeff_X_pow, if_neg, mul_zero]
    /-
      case neg.hnc
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      n d : Nat
      h : Not (LE.le n d)
      x : Prod Nat Nat
      hx : Membership.mem (Finset.HasAntidiagonal.antidiagonal d) x
      ⊢ Not (Eq x.2 n)
    -/
    exact ((le_of_add_le_right (mem_antidiagonal.mp hx).le).trans_lt <| not_le.mp h).ne
    /-
      🎉 no goals
    -/


theorem coeff_X_pow_mul' (p : R[X]) (n d : ℕ) :
    (X ^ n * p).coeff d = ite (n ≤ d) (p.coeff (d - n)) 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n d : Nat
    ⊢ Eq ((HMul.hMul (HPow.hPow Polynomial.X n) p).coeff d) (ite (LE.le n d) (p.co …
  -/
  rw [(commute_X_pow p n).eq, coeff_mul_X_pow']
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_mul_X (p : R[X]) (n : ℕ) : coeff (p * X) (n + 1) = coeff p n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq ((HMul.hMul p Polynomial.X).coeff (HAdd.hAdd n 1)) (p.coeff n)
  -/
  simpa only [pow_one] using coeff_mul_X_pow p 1 n
  /-
    🎉 no goals
  -/


@[simp]
theorem coeff_X_mul (p : R[X]) (n : ℕ) : coeff (X * p) (n + 1) = coeff p n := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    ⊢ Eq ((HMul.hMul Polynomial.X p).coeff (HAdd.hAdd n 1)) (p.coeff n)
  -/
  rw [(commute_X p).eq, coeff_mul_X]
  /-
    🎉 no goals
  -/


theorem coeff_mul_monomial (p : R[X]) (n d : ℕ) (r : R) :
    coeff (p * monomial n r) (d + n) = coeff p d * r := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n d : Nat
    r : R
    ⊢ Eq ((HMul.hMul p ((Polynomial.monomial n) r)).coeff (HAdd.hAdd d n)) (HMul.h …
  -/
  rw [← C_mul_X_pow_eq_monomial, ← X_pow_mul, ← mul_assoc, coeff_mul_C, coeff_mul_X_pow]
  /-
    🎉 no goals
  -/


theorem coeff_monomial_mul (p : R[X]) (n d : ℕ) (r : R) :
    coeff (monomial n r * p) (d + n) = r * coeff p d := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n d : Nat
    r : R
    ⊢ Eq ((HMul.hMul ((Polynomial.monomial n) r) p).coeff (HAdd.hAdd d n)) (HMul.h …
  -/
  rw [← C_mul_X_pow_eq_monomial, mul_assoc, coeff_C_mul, X_pow_mul, coeff_mul_X_pow]
  /-
    🎉 no goals
  -/

-- This can already be proved by `simp`.

theorem coeff_mul_monomial_zero (p : R[X]) (d : ℕ) (r : R) :
    coeff (p * monomial 0 r) d = coeff p d * r :=
  coeff_mul_monomial p 0 d r

-- This can already be proved by `simp`.

theorem coeff_monomial_zero_mul (p : R[X]) (d : ℕ) (r : R) :
    coeff (monomial 0 r * p) d = r * coeff p d :=
  coeff_monomial_mul p 0 d r


theorem mul_X_pow_eq_zero {p : R[X]} {n : ℕ} (H : p * X ^ n = 0) : p = 0 :=
  ext fun k => (coeff_mul_X_pow p n k).symm.trans <| ext_iff.1 H (k + n)


theorem isRegular_X_pow (n : ℕ) : IsRegular (X ^ n : R[X]) := by
  suffices IsLeftRegular (X^n : R[X]) from
    ⟨this, this.right_of_commute (fun p => commute_X_pow p n)⟩
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    ⊢ IsLeftRegular (HPow.hPow Polynomial.X n)
  -/
  intro P Q (hPQ : X^n * P = X^n * Q)
  /-
    R : Type u
    inst✝ : Semiring R
    n : Nat
    P Q : Polynomial R
    hPQ : Eq (HMul.hMul (HPow.hPow Polynomial.X n) P) (HMul.hMul (HPow.hPow Polyno …
    ⊢ Eq P Q
  -/
  ext i
  /-
    case a
    R : Type u
    inst✝ : Semiring R
    n : Nat
    P Q : Polynomial R
    hPQ : Eq (HMul.hMul (HPow.hPow Polynomial.X n) P) (HMul.hMul (HPow.hPow Polyno …
    i : Nat
    ⊢ Eq (P.coeff i) (Q.coeff i)
  -/
  rw [← coeff_X_pow_mul P n i, hPQ, coeff_X_pow_mul Q n i]
  /-
    🎉 no goals
  -/


@[simp] theorem isRegular_X : IsRegular (X : R[X]) := pow_one (X : R[X]) ▸ isRegular_X_pow 1


theorem coeff_X_add_C_pow (r : R) (n k : ℕ) :
    ((X + C r) ^ n).coeff k = r ^ (n - k) * (n.choose k : R) := by
  /-
    R : Type u
    inst✝ : Semiring R
    r : R
    n k : Nat
    ⊢ Eq ((HPow.hPow (HAdd.hAdd Polynomial.X (Polynomial.C r)) n).coeff k) (HMul.h …
  -/
  rw [(commute_X (C r : R[X])).add_pow, ← lcoeff_apply, map_sum]
  /-
    R : Type u
    inst✝ : Semiring R
    r : R
    n k : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => (Polynomial.lcoeff R k) (HMu …
  -/
  simp only [one_pow, mul_one, lcoeff_apply, ← C_eq_natCast, ← C_pow, coeff_mul_C, Nat.cast_id]
  /-
    R : Type u
    inst✝ : Semiring R
    r : R
    n k : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd n 1)).sum fun x => HMul.hMul (HMul.hMul ((HPow. …
  -/
  rw [Finset.sum_eq_single k, coeff_X_pow_self, one_mul]
    /-
      case h₀
      R : Type u
      inst✝ : Semiring R
      r : R
      n k : Nat
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd n 1)) b → Ne b k → Eq ( …
    -/
  · intro _ _ h
    /-
      case h₀
      R : Type u
      inst✝ : Semiring R
      r : R
      n k b✝ : Nat
      a✝ : Membership.mem (Finset.range (HAdd.hAdd n 1)) b✝
      h : Ne b✝ k
      ⊢ Eq (HMul.hMul (HMul.hMul ((HPow.hPow Polynomial.X b✝).coeff k) (HPow.hPow r  …
    -/
    simp [coeff_X_pow, h.symm]
    /-
      🎉 no goals
    -/
    /-
      case h₁
      R : Type u
      inst✝ : Semiring R
      r : R
      n k : Nat
      ⊢ Not (Membership.mem (Finset.range (HAdd.hAdd n 1)) k) → Eq (HMul.hMul (HMul. …
    -/
  · simp only [coeff_X_pow_self, one_mul, not_lt, Finset.mem_range]
    /-
      case h₁
      R : Type u
      inst✝ : Semiring R
      r : R
      n k : Nat
      ⊢ LE.le (HAdd.hAdd n 1) k → Eq (HMul.hMul (HPow.hPow r (HSub.hSub n k)) ↑(n.ch …
    -/
    intro h
    /-
      case h₁
      R : Type u
      inst✝ : Semiring R
      r : R
      n k : Nat
      h : LE.le (HAdd.hAdd n 1) k
      ⊢ Eq (HMul.hMul (HPow.hPow r (HSub.hSub n k)) ↑(n.choose k)) 0
    -/
    rw [Nat.choose_eq_zero_of_lt h, Nat.cast_zero, mul_zero]
    /-
      🎉 no goals
    -/


theorem coeff_X_add_one_pow (R : Type*) [Semiring R] (n k : ℕ) :
                                                   /-
                                                     R : Type u_1
                                                     inst✝ : Semiring R
                                                     n k : Nat
                                                     ⊢ Eq ((HPow.hPow (HAdd.hAdd Polynomial.X 1) n).coeff k) ↑(n.choose k)
                                                   -/
    ((X + 1) ^ n).coeff k = (n.choose k : R) := by rw [← C_1, coeff_X_add_C_pow, one_pow, one_mul]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem coeff_one_add_X_pow (R : Type*) [Semiring R] (n k : ℕ) :
                                                   /-
                                                     R : Type u_1
                                                     inst✝ : Semiring R
                                                     n k : Nat
                                                     ⊢ Eq ((HPow.hPow (HAdd.hAdd 1 Polynomial.X) n).coeff k) ↑(n.choose k)
                                                   -/
    ((1 + X) ^ n).coeff k = (n.choose k : R) := by rw [add_comm _ X, coeff_X_add_one_pow]
                                                   /-
                                                     🎉 no goals
                                                   -/


theorem C_dvd_iff_dvd_coeff (r : R) (φ : R[X]) : C r ∣ φ ↔ ∀ i, r ∣ φ.coeff i := by
  /-
    R : Type u
    inst✝ : Semiring R
    r : R
    φ : Polynomial R
    ⊢ Iff (Dvd.dvd (Polynomial.C r) φ) (∀ (i : Nat), Dvd.dvd r (φ.coeff i))
  -/
  constructor
    /-
      case mp
      R : Type u
      inst✝ : Semiring R
      r : R
      φ : Polynomial R
      ⊢ Dvd.dvd (Polynomial.C r) φ → ∀ (i : Nat), Dvd.dvd r (φ.coeff i)
    -/
  · rintro ⟨φ, rfl⟩ c
    /-
      case mp.intro
      R : Type u
      inst✝ : Semiring R
      r : R
      φ : Polynomial R
      c : Nat
      ⊢ Dvd.dvd r ((HMul.hMul (Polynomial.C r) φ).coeff c)
    -/
    rw [coeff_C_mul]
    /-
      case mp.intro
      R : Type u
      inst✝ : Semiring R
      r : R
      φ : Polynomial R
      c : Nat
      ⊢ Dvd.dvd r (HMul.hMul r (φ.coeff c))
    -/
    apply dvd_mul_right
    /-
      🎉 no goals
    -/
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      r : R
      φ : Polynomial R
      ⊢ (∀ (i : Nat), Dvd.dvd r (φ.coeff i)) → Dvd.dvd (Polynomial.C r) φ
    -/
  · intro h
    /-
      case mpr
      R : Type u
      inst✝ : Semiring R
      r : R
      φ : Polynomial R
      h : ∀ (i : Nat), Dvd.dvd r (φ.coeff i)
      ⊢ Dvd.dvd (Polynomial.C r) φ
    -/
    choose c hc using h
    classical
      let c' : ℕ → R := fun i => if i ∈ φ.support then c i else 0
      let ψ : R[X] := ∑ i ∈ φ.support, monomial i (c' i)
      use ψ
      ext i
      simp only [c', ψ, coeff_C_mul, mem_support_iff, coeff_monomial, finset_sum_coeff,
        Finset.sum_ite_eq']
      split_ifs with hi
      · rw [hc]
      · rw [Classical.not_not] at hi
        rwa [mul_zero]


                                                      /-
                                                        R : Type u
                                                        inst✝ : Semiring R
                                                        p : Polynomial R
                                                        a : R
                                                        ⊢ Eq (HSMul.hSMul a p) (HMul.hMul (Polynomial.C a) p)
                                                      -/
theorem smul_eq_C_mul (a : R) : a • p = C a * p := by simp [ext_iff]
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem update_eq_add_sub_coeff {R : Type*} [Ring R] (p : R[X]) (n : ℕ) (a : R) :
    p.update n a = p + Polynomial.C (a - p.coeff n) * Polynomial.X ^ n := by
  /-
    R : Type u_1
    inst✝ : Ring R
    p : Polynomial R
    n : Nat
    a : R
    ⊢ Eq (p.update n a) (HAdd.hAdd p (HMul.hMul (Polynomial.C (HSub.hSub a (p.coef …
  -/
  ext
  /-
    case a
    R : Type u_1
    inst✝ : Ring R
    p : Polynomial R
    n : Nat
    a : R
    n✝ : Nat
    ⊢ Eq ((p.update n a).coeff n✝) ((HAdd.hAdd p (HMul.hMul (Polynomial.C (HSub.hS …
  -/
  rw [coeff_update_apply, coeff_add, coeff_C_mul_X_pow]
  /-
    case a
    R : Type u_1
    inst✝ : Ring R
    p : Polynomial R
    n : Nat
    a : R
    n✝ : Nat
    ⊢ Eq (ite (Eq n✝ n) a (p.coeff n✝)) (HAdd.hAdd (p.coeff n✝) (ite (Eq n✝ n) (HS …
  -/
                       /-
                         🎉 no goals
                       -/
  split_ifs with h <;> simp [h]
                       /-
                         🎉 no goals
                       -/


theorem natCast_coeff_zero {n : ℕ} {R : Type*} [Semiring R] : (n : R[X]).coeff 0 = n := by
  /-
    n : Nat
    R : Type u_1
    inst✝ : Semiring R
    ⊢ Eq ((↑n).coeff 0) ↑n
  -/
  simp only [coeff_natCast_ite, ite_true]
  /-
    🎉 no goals
  -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_coeff_zero := natCast_coeff_zero


@[norm_cast]
theorem natCast_inj {m n : ℕ} {R : Type*} [Semiring R] [CharZero R] :
    (↑m : R[X]) = ↑n ↔ m = n := by
  /-
    m n : Nat
    R : Type u_1
    inst✝¹ : Semiring R
    inst✝ : CharZero R
    ⊢ Iff (Eq ↑m ↑n) (Eq m n)
  -/
  constructor
    /-
      case mp
      m n : Nat
      R : Type u_1
      inst✝¹ : Semiring R
      inst✝ : CharZero R
      ⊢ Eq ↑m ↑n → Eq m n
    -/
  · intro h
    /-
      case mp
      m n : Nat
      R : Type u_1
      inst✝¹ : Semiring R
      inst✝ : CharZero R
      h : Eq ↑m ↑n
      ⊢ Eq m n
    -/
    apply_fun fun p => p.coeff 0 at h
    /-
      case mp
      m n : Nat
      R : Type u_1
      inst✝¹ : Semiring R
      inst✝ : CharZero R
      h : Eq ((↑m).coeff 0) ((↑n).coeff 0)
      ⊢ Eq m n
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      m n : Nat
      R : Type u_1
      inst✝¹ : Semiring R
      inst✝ : CharZero R
      ⊢ Eq m n → Eq ↑m ↑n
    -/
  · rintro rfl
    /-
      case mpr
      m : Nat
      R : Type u_1
      inst✝¹ : Semiring R
      inst✝ : CharZero R
      ⊢ Eq ↑m ↑m
    -/
    rfl
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias nat_cast_inj := natCast_inj


@[simp]
theorem intCast_coeff_zero {i : ℤ} {R : Type*} [Ring R] : (i : R[X]).coeff 0 = i := by
  /-
    i : Int
    R : Type u_1
    inst✝ : Ring R
    ⊢ Eq ((↑i).coeff 0) ↑i
  -/
              /-
                🎉 no goals
              -/
  cases i <;> simp
              /-
                🎉 no goals
              -/


@[deprecated (since := "2024-04-17")]
alias int_cast_coeff_zero := intCast_coeff_zero


@[norm_cast]
theorem intCast_inj {m n : ℤ} {R : Type*} [Ring R] [CharZero R] : (↑m : R[X]) = ↑n ↔ m = n := by
  /-
    m n : Int
    R : Type u_1
    inst✝¹ : Ring R
    inst✝ : CharZero R
    ⊢ Iff (Eq ↑m ↑n) (Eq m n)
  -/
  constructor
    /-
      case mp
      m n : Int
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharZero R
      ⊢ Eq ↑m ↑n → Eq m n
    -/
  · intro h
    /-
      case mp
      m n : Int
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharZero R
      h : Eq ↑m ↑n
      ⊢ Eq m n
    -/
    apply_fun fun p => p.coeff 0 at h
    /-
      case mp
      m n : Int
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharZero R
      h : Eq ((↑m).coeff 0) ((↑n).coeff 0)
      ⊢ Eq m n
    -/
    simpa using h
    /-
      🎉 no goals
    -/
    /-
      case mpr
      m n : Int
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharZero R
      ⊢ Eq m n → Eq ↑m ↑n
    -/
  · rintro rfl
    /-
      case mpr
      m : Int
      R : Type u_1
      inst✝¹ : Ring R
      inst✝ : CharZero R
      ⊢ Eq ↑m ↑m
    -/
    rfl
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-04-17")]
alias int_cast_inj := intCast_inj


instance charZero [CharZero R] : CharZero R[X] where cast_injective _x _y := natCast_inj.mp


