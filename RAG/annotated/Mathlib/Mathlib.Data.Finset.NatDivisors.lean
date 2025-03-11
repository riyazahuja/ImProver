/-- The divisors of a product of natural numbers are the pointwise product of the divisors of the
factors. -/
lemma Nat.divisors_mul (m n : ℕ) : divisors (m * n) = divisors m * divisors n := by
  /-
    m n : Nat
    ⊢ Eq (HMul.hMul m n).divisors (HMul.hMul m.divisors n.divisors)
  -/
  ext k
  /-
    case h
    m n k : Nat
    ⊢ Iff (Membership.mem (HMul.hMul m n).divisors k) (Membership.mem (HMul.hMul m …
  -/
  simp_rw [mem_mul, mem_divisors, dvd_mul, mul_ne_zero_iff, ← exists_and_left, ← exists_and_right]
  /-
    case h
    m n k : Nat
    ⊢ Iff (Exists fun x => Exists fun x_1 => And (And (Dvd.dvd x m) (And (Dvd.dvd  …
  -/
  simp only [and_assoc, and_comm, and_left_comm]
  /-
    🎉 no goals
  -/


/-- `Nat.divisors` as a `MonoidHom`. -/
@[simps]
def Nat.divisorsHom : ℕ →* Finset ℕ where
  toFun := Nat.divisors
  map_mul' := divisors_mul
  map_one' := divisors_one


lemma Nat.Prime.divisors_sq {p : ℕ} (hp : p.Prime) : (p ^ 2).divisors = {p ^ 2, p, 1} := by
  /-
    p : Nat
    hp : Nat.Prime p
    ⊢ Eq (HPow.hPow p 2).divisors (Insert.insert (HPow.hPow p 2) (Insert.insert p  …
  -/
  simp [divisors_prime_pow hp, range_succ]
  /-
    🎉 no goals
  -/


lemma List.nat_divisors_prod (l : List ℕ) : divisors l.prod = (l.map divisors).prod :=
  map_list_prod Nat.divisorsHom l


lemma Multiset.nat_divisors_prod (s : Multiset ℕ) : divisors s.prod = (s.map divisors).prod :=
  map_multiset_prod Nat.divisorsHom s


lemma Finset.nat_divisors_prod {ι : Type*} (s : Finset ι) (f : ι → ℕ) :
    divisors (∏ i ∈ s, f i) = ∏ i ∈ s, divisors (f i) :=
  map_prod Nat.divisorsHom f s

