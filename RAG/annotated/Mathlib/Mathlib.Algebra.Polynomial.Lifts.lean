/-- We define the subsemiring of polynomials that lifts as the image of `RingHom.of (map f)`. -/
def lifts (f : R →+* S) : Subsemiring S[X] :=
  RingHom.rangeS (mapRingHom f)


theorem mem_lifts (p : S[X]) : p ∈ lifts f ↔ ∃ q : R[X], map f q = p := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    ⊢ Iff (Membership.mem (Polynomial.lifts f) p) (Exists fun q => Eq (Polynomial. …
  -/
  simp only [coe_mapRingHom, lifts, RingHom.mem_rangeS]
  /-
    🎉 no goals
  -/


theorem lifts_iff_set_range (p : S[X]) : p ∈ lifts f ↔ p ∈ Set.range (map f) := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    ⊢ Iff (Membership.mem (Polynomial.lifts f) p) (Membership.mem (Set.range (Poly …
  -/
  simp only [coe_mapRingHom, lifts, Set.mem_range, RingHom.mem_rangeS]
  /-
    🎉 no goals
  -/


theorem lifts_iff_ringHom_rangeS (p : S[X]) : p ∈ lifts f ↔ p ∈ (mapRingHom f).rangeS := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    ⊢ Iff (Membership.mem (Polynomial.lifts f) p) (Membership.mem (Polynomial.mapR …
  -/
  simp only [coe_mapRingHom, lifts, Set.mem_range, RingHom.mem_rangeS]
  /-
    🎉 no goals
  -/


theorem lifts_iff_coeff_lifts (p : S[X]) : p ∈ lifts f ↔ ∀ n : ℕ, p.coeff n ∈ Set.range f := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    ⊢ Iff (Membership.mem (Polynomial.lifts f) p) (∀ (n : Nat), Membership.mem (Se …
  -/
  rw [lifts_iff_ringHom_rangeS, mem_map_rangeS f]
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    ⊢ Iff (∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)) (∀ (n : Nat), Members …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- If `(r : R)`, then `C (f r)` lifts. -/
theorem C_mem_lifts (f : R →+* S) (r : R) : C (f r) ∈ lifts f :=
  ⟨C r, by
    simp only [coe_mapRingHom, map_C, Set.mem_univ, Subsemiring.coe_top, eq_self_iff_true,
      and_self_iff]⟩


/-- If `(s : S)` is in the image of `f`, then `C s` lifts. -/
theorem C'_mem_lifts {f : R →+* S} {s : S} (h : s ∈ Set.range f) : C s ∈ lifts f := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    s : S
    h : Membership.mem (Set.range ⇑f) s
    ⊢ Membership.mem (Polynomial.lifts f) (Polynomial.C s)
  -/
  obtain ⟨r, rfl⟩ := Set.mem_range.1 h
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    r : R
    h : Membership.mem (Set.range ⇑f) (f r)
    ⊢ Membership.mem (Polynomial.lifts f) (Polynomial.C (f r))
  -/
  use C r
  simp only [coe_mapRingHom, map_C, Set.mem_univ, Subsemiring.coe_top, eq_self_iff_true,
    and_self_iff]


/-- The polynomial `X` lifts. -/
theorem X_mem_lifts (f : R →+* S) : (X : S[X]) ∈ lifts f :=
  ⟨X, by
    simp only [coe_mapRingHom, Set.mem_univ, Subsemiring.coe_top, eq_self_iff_true, map_X,
      and_self_iff]⟩


/-- The polynomial `X ^ n` lifts. -/
theorem X_pow_mem_lifts (f : R →+* S) (n : ℕ) : (X ^ n : S[X]) ∈ lifts f :=
  ⟨X ^ n, by
    simp only [coe_mapRingHom, map_pow, Set.mem_univ, Subsemiring.coe_top, eq_self_iff_true,
      map_X, and_self_iff]⟩


/-- If `p` lifts and `(r : R)` then `r * p` lifts. -/
theorem base_mul_mem_lifts {p : S[X]} (r : R) (hp : p ∈ lifts f) : C (f r) * p ∈ lifts f := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    r : R
    hp : Membership.mem (Polynomial.lifts f) p
    ⊢ Membership.mem (Polynomial.lifts f) (HMul.hMul (Polynomial.C (f r)) p)
  -/
  simp only [lifts, RingHom.mem_rangeS] at hp ⊢
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    r : R
    hp : Exists fun x => Eq ((Polynomial.mapRingHom f) x) p
    ⊢ Exists fun x => Eq ((Polynomial.mapRingHom f) x) (HMul.hMul (Polynomial.C (f …
  -/
  obtain ⟨p₁, rfl⟩ := hp
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    r : R
    p₁ : Polynomial R
    ⊢ Exists fun x => Eq ((Polynomial.mapRingHom f) x) (HMul.hMul (Polynomial.C (f …
  -/
  use C r * p₁
  /-
    case h
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    r : R
    p₁ : Polynomial R
    ⊢ Eq ((Polynomial.mapRingHom f) (HMul.hMul (Polynomial.C r) p₁)) (HMul.hMul (P …
  -/
  simp only [coe_mapRingHom, map_C, map_mul]
  /-
    🎉 no goals
  -/


/-- If `(s : S)` is in the image of `f`, then `monomial n s` lifts. -/
theorem monomial_mem_lifts {s : S} (n : ℕ) (h : s ∈ Set.range f) : monomial n s ∈ lifts f := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    s : S
    n : Nat
    h : Membership.mem (Set.range ⇑f) s
    ⊢ Membership.mem (Polynomial.lifts f) ((Polynomial.monomial n) s)
  -/
  obtain ⟨r, rfl⟩ := Set.mem_range.1 h
  /-
    case intro
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    r : R
    h : Membership.mem (Set.range ⇑f) (f r)
    ⊢ Membership.mem (Polynomial.lifts f) ((Polynomial.monomial n) (f r))
  -/
  use monomial n r
  simp only [coe_mapRingHom, Set.mem_univ, map_monomial, Subsemiring.coe_top, eq_self_iff_true,
    and_self_iff]


/-- If `p` lifts then `p.erase n` lifts. -/
theorem erase_mem_lifts {p : S[X]} (n : ℕ) (h : p ∈ lifts f) : p.erase n ∈ lifts f := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    n : Nat
    h : Membership.mem (Polynomial.lifts f) p
    ⊢ Membership.mem (Polynomial.lifts f) (Polynomial.erase n p)
  -/
  rw [lifts_iff_ringHom_rangeS, mem_map_rangeS] at h ⊢
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    n : Nat
    h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
    ⊢ ∀ (n_1 : Nat), Membership.mem f.rangeS ((Polynomial.erase n p).coeff n_1)
  -/
  intro k
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    n : Nat
    h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
    k : Nat
    ⊢ Membership.mem f.rangeS ((Polynomial.erase n p).coeff k)
  -/
  by_cases hk : k = n
    /-
      case pos
      R : Type u
      inst✝¹ : Semiring R
      S : Type v
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      n : Nat
      h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
      k : Nat
      hk : Eq k n
      ⊢ Membership.mem f.rangeS ((Polynomial.erase n p).coeff k)
    -/
  · use 0
    /-
      case h
      R : Type u
      inst✝¹ : Semiring R
      S : Type v
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      n : Nat
      h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
      k : Nat
      hk : Eq k n
      ⊢ Eq (f 0) ((Polynomial.erase n p).coeff k)
    -/
    simp only [hk, RingHom.map_zero, erase_same]
    /-
      🎉 no goals
    -/
  /-
    case neg
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    n : Nat
    h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
    k : Nat
    hk : Not (Eq k n)
    ⊢ Membership.mem f.rangeS ((Polynomial.erase n p).coeff k)
  -/
  obtain ⟨i, hi⟩ := h k
  /-
    case neg.intro
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    n : Nat
    h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
    k : Nat
    hk : Not (Eq k n)
    i : R
    hi : Eq (f i) (p.coeff k)
    ⊢ Membership.mem f.rangeS ((Polynomial.erase n p).coeff k)
  -/
  use i
  /-
    case h
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    n : Nat
    h : ∀ (n : Nat), Membership.mem f.rangeS (p.coeff n)
    k : Nat
    hk : Not (Eq k n)
    i : R
    hi : Eq (f i) (p.coeff k)
    ⊢ Eq (f i) ((Polynomial.erase n p).coeff k)
  -/
  simp only [hi, hk, erase_ne, Ne, not_false_iff]
  /-
    🎉 no goals
  -/


theorem monomial_mem_lifts_and_degree_eq {s : S} {n : ℕ} (hl : monomial n s ∈ lifts f) :
    ∃ q : R[X], map f q = monomial n s ∧ q.degree = (monomial n s).degree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    s : S
    n : Nat
    hl : Membership.mem (Polynomial.lifts f) ((Polynomial.monomial n) s)
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) ((Polynomial.monomial n) s)) (E …
  -/
  rcases eq_or_ne s 0 with rfl | h
    /-
      case inl
      R : Type u
      inst✝¹ : Semiring R
      S : Type v
      inst✝ : Semiring S
      f : RingHom R S
      n : Nat
      hl : Membership.mem (Polynomial.lifts f) ((Polynomial.monomial n) 0)
      ⊢ Exists fun q => And (Eq (Polynomial.map f q) ((Polynomial.monomial n) 0)) (E …
    -/
  · exact ⟨0, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    s : S
    n : Nat
    hl : Membership.mem (Polynomial.lifts f) ((Polynomial.monomial n) s)
    h : Ne s 0
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) ((Polynomial.monomial n) s)) (E …
  -/
  obtain ⟨a, rfl⟩ := coeff_monomial_same n s ▸ (monomial n s).lifts_iff_coeff_lifts.mp hl n
  /-
    case inr.intro
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    a : R
    hl : Membership.mem (Polynomial.lifts f) ((Polynomial.monomial n) (f a))
    h : Ne (f a) 0
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) ((Polynomial.monomial n) (f a)) …
  -/
  refine ⟨monomial n a, map_monomial f, ?_⟩
  /-
    case inr.intro
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    a : R
    hl : Membership.mem (Polynomial.lifts f) ((Polynomial.monomial n) (f a))
    h : Ne (f a) 0
    ⊢ Eq ((Polynomial.monomial n) a).degree ((Polynomial.monomial n) (f a)).degree
  -/
  rw [degree_monomial, degree_monomial n h]
  /-
    case inr.intro.ha
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    n : Nat
    a : R
    hl : Membership.mem (Polynomial.lifts f) ((Polynomial.monomial n) (f a))
    h : Ne (f a) 0
    ⊢ Ne a 0
  -/
  exact mt (fun ha ↦ ha ▸ map_zero f) h
  /-
    🎉 no goals
  -/


/-- A polynomial lifts if and only if it can be lifted to a polynomial of the same degree. -/
theorem mem_lifts_and_degree_eq {p : S[X]} (hlifts : p ∈ lifts f) :
    ∃ q : R[X], map f q = p ∧ q.degree = p.degree := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : Membership.mem (Polynomial.lifts f) p
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (Eq q.degree p.degree)
  -/
  rw [lifts_iff_coeff_lifts] at hlifts
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (Eq q.degree p.degree)
  -/
  let g : ℕ → R := fun k ↦ (hlifts k).choose
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    g : Nat → R := fun k => Exists.choose ⋯
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (Eq q.degree p.degree)
  -/
  have hg : ∀ k, f (g k) = p.coeff k := fun k ↦ (hlifts k).choose_spec
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    g : Nat → R := fun k => Exists.choose ⋯
    hg : ∀ (k : Nat), Eq (f (g k)) (p.coeff k)
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (Eq q.degree p.degree)
  -/
  let q : R[X] := ∑ k ∈ p.support, monomial k (g k)
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    g : Nat → R := fun k => Exists.choose ⋯
    hg : ∀ (k : Nat), Eq (f (g k)) (p.coeff k)
    q : Polynomial R := p.support.sum fun k => (Polynomial.monomial k) (g k)
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (Eq q.degree p.degree)
  -/
  have hq : map f q = p := by simp_rw [q, Polynomial.map_sum, map_monomial, hg, ← as_sum_support]
  have hq' : q.support = p.support := by
    simp_rw [Finset.ext_iff, mem_support_iff, q, finset_sum_coeff, coeff_monomial,
      Finset.sum_ite_eq', ite_ne_right_iff, mem_support_iff, and_iff_left_iff_imp, not_imp_not]
    exact fun k h ↦ by rw [← hg, h, map_zero]
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    g : Nat → R := fun k => Exists.choose ⋯
    hg : ∀ (k : Nat), Eq (f (g k)) (p.coeff k)
    q : Polynomial R := p.support.sum fun k => (Polynomial.monomial k) (g k)
    hq : Eq (Polynomial.map f q) p
    hq' : Eq q.support p.support
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (Eq q.degree p.degree)
  -/
  exact ⟨q, hq, congrArg Finset.max hq'⟩
  /-
    🎉 no goals
  -/


/-- A monic polynomial lifts if and only if it can be lifted to a monic polynomial
of the same degree. -/
theorem lifts_and_degree_eq_and_monic [Nontrivial S] {p : S[X]} (hlifts : p ∈ lifts f)
    (hp : p.Monic) : ∃ q : R[X], map f q = p ∧ q.degree = p.degree ∧ q.Monic := by
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type v
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Nontrivial S
    p : Polynomial S
    hlifts : Membership.mem (Polynomial.lifts f) p
    hp : p.Monic
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.degree p.degree)  …
  -/
  rw [lifts_iff_coeff_lifts] at hlifts
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type v
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Nontrivial S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    hp : p.Monic
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.degree p.degree)  …
  -/
  let g : ℕ → R := fun k ↦ (hlifts k).choose
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type v
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Nontrivial S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    hp : p.Monic
    g : Nat → R := fun k => Exists.choose ⋯
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.degree p.degree)  …
  -/
  have hg k : f (g k) = p.coeff k := (hlifts k).choose_spec
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type v
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Nontrivial S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    hp : p.Monic
    g : Nat → R := fun k => Exists.choose ⋯
    hg : ∀ (k : Nat), Eq (f (g k)) (p.coeff k)
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.degree p.degree)  …
  -/
  let q : R[X] := X ^ p.natDegree + ∑ k ∈ Finset.range p.natDegree, C (g k) * X ^ k
  have hq : map f q = p := by
    simp_rw [q, Polynomial.map_add, Polynomial.map_sum, Polynomial.map_mul, Polynomial.map_pow,
      map_X, map_C, hg, ← hp.as_sum]
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type v
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Nontrivial S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    hp : p.Monic
    g : Nat → R := fun k => Exists.choose ⋯
    hg : ∀ (k : Nat), Eq (f (g k)) (p.coeff k)
    q : Polynomial R := HAdd.hAdd (HPow.hPow Polynomial.X p.natDegree) ((Finset.ra …
    hq : Eq (Polynomial.map f q) p
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.degree p.degree)  …
  -/
  have h : q.Monic := monic_X_pow_add (by simp_rw [← Fin.sum_univ_eq_sum_range, degree_sum_fin_lt])
  /-
    R : Type u
    inst✝² : Semiring R
    S : Type v
    inst✝¹ : Semiring S
    f : RingHom R S
    inst✝ : Nontrivial S
    p : Polynomial S
    hlifts : ∀ (n : Nat), Membership.mem (Set.range ⇑f) (p.coeff n)
    hp : p.Monic
    g : Nat → R := fun k => Exists.choose ⋯
    hg : ∀ (k : Nat), Eq (f (g k)) (p.coeff k)
    q : Polynomial R := HAdd.hAdd (HPow.hPow Polynomial.X p.natDegree) ((Finset.ra …
    hq : Eq (Polynomial.map f q) p
    h : q.Monic
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.degree p.degree)  …
  -/
  exact ⟨q, hq, hq ▸ (h.degree_map f).symm, h⟩
  /-
    🎉 no goals
  -/


theorem lifts_and_natDegree_eq_and_monic {p : S[X]} (hlifts : p ∈ lifts f) (hp : p.Monic) :
    ∃ q : R[X], map f q = p ∧ q.natDegree = p.natDegree ∧ q.Monic := by
  /-
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : Membership.mem (Polynomial.lifts f) p
    hp : p.Monic
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.natDegree p.natDe …
  -/
  cases' subsingleton_or_nontrivial S with hR hR
    /-
      case inl
      R : Type u
      inst✝¹ : Semiring R
      S : Type v
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial S
      hlifts : Membership.mem (Polynomial.lifts f) p
      hp : p.Monic
      hR : Subsingleton S
      ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.natDegree p.natDe …
    -/
  · obtain rfl : p = 1 := Subsingleton.elim _ _
    /-
      case inl
      R : Type u
      inst✝¹ : Semiring R
      S : Type v
      inst✝ : Semiring S
      f : RingHom R S
      hR : Subsingleton S
      hlifts : Membership.mem (Polynomial.lifts f) 1
      hp : Polynomial.Monic 1
      ⊢ Exists fun q => And (Eq (Polynomial.map f q) 1) (And (Eq q.natDegree (Polyno …
    -/
    exact ⟨1, Subsingleton.elim _ _, by simp, by simp⟩
    /-
      🎉 no goals
    -/
  /-
    case inr
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : Membership.mem (Polynomial.lifts f) p
    hp : p.Monic
    hR : Nontrivial S
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.natDegree p.natDe …
  -/
  obtain ⟨p', h₁, h₂, h₃⟩ := lifts_and_degree_eq_and_monic hlifts hp
  /-
    case inr.intro.intro.intro
    R : Type u
    inst✝¹ : Semiring R
    S : Type v
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial S
    hlifts : Membership.mem (Polynomial.lifts f) p
    hp : p.Monic
    hR : Nontrivial S
    p' : Polynomial R
    h₁ : Eq (Polynomial.map f p') p
    h₂ : Eq p'.degree p.degree
    h₃ : p'.Monic
    ⊢ Exists fun q => And (Eq (Polynomial.map f q) p) (And (Eq q.natDegree p.natDe …
  -/
  exact ⟨p', h₁, natDegree_eq_of_degree_eq h₂, h₃⟩
  /-
    🎉 no goals
  -/


/-- The subring of polynomials that lift. -/
def liftsRing (f : R →+* S) : Subring S[X] :=
  RingHom.range (mapRingHom f)


/-- If `R` and `S` are rings, `p` is in the subring of polynomials that lift if and only if it is in
the subsemiring of polynomials that lift. -/
theorem lifts_iff_liftsRing (p : S[X]) : p ∈ lifts f ↔ p ∈ liftsRing f := by
  /-
    R : Type u
    inst✝¹ : Ring R
    S : Type v
    inst✝ : Ring S
    f : RingHom R S
    p : Polynomial S
    ⊢ Iff (Membership.mem (Polynomial.lifts f) p) (Membership.mem (Polynomial.lift …
  -/
  simp only [lifts, liftsRing, RingHom.mem_range, RingHom.mem_rangeS]
  /-
    🎉 no goals
  -/


/-- The map `R[X] → S[X]` as an algebra homomorphism. -/
def mapAlg (R : Type u) [CommSemiring R] (S : Type v) [Semiring S] [Algebra R S] :
    R[X] →ₐ[R] S[X] :=
  @aeval _ S[X] _ _ _ (X : S[X])


/-- `mapAlg` is the morphism induced by `R → S`. -/
theorem mapAlg_eq_map (p : R[X]) : mapAlg R S p = map (algebraMap R S) p := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial R
    ⊢ Eq ((Polynomial.mapAlg R S) p) (Polynomial.map (algebraMap R S) p)
  -/
  simp only [mapAlg, aeval_def, eval₂_eq_sum, map, algebraMap_apply, RingHom.coe_comp]
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial R
    ⊢ Eq (p.sum fun e a => HMul.hMul (Polynomial.C ((algebraMap R S) a)) (HPow.hPo …
  -/
  ext; congr
       /-
         🎉 no goals
       -/


/-- A polynomial `p` lifts if and only if it is in the image of `mapAlg`. -/
theorem mem_lifts_iff_mem_alg (R : Type u) [CommSemiring R] {S : Type v} [Semiring S] [Algebra R S]
    (p : S[X]) : p ∈ lifts (algebraMap R S) ↔ p ∈ AlgHom.range (@mapAlg R _ S _ _) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial S
    ⊢ Iff (Membership.mem (Polynomial.lifts (algebraMap R S)) p) (Membership.mem ( …
  -/
  simp only [coe_mapRingHom, lifts, mapAlg_eq_map, AlgHom.mem_range, RingHom.mem_rangeS]
  /-
    🎉 no goals
  -/


/-- If `p` lifts and `(r : R)` then `r • p` lifts. -/
theorem smul_mem_lifts {p : S[X]} (r : R) (hp : p ∈ lifts (algebraMap R S)) :
    r • p ∈ lifts (algebraMap R S) := by
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial S
    r : R
    hp : Membership.mem (Polynomial.lifts (algebraMap R S)) p
    ⊢ Membership.mem (Polynomial.lifts (algebraMap R S)) (HSMul.hSMul r p)
  -/
  rw [mem_lifts_iff_mem_alg] at hp ⊢
  /-
    R : Type u
    inst✝² : CommSemiring R
    S : Type v
    inst✝¹ : Semiring S
    inst✝ : Algebra R S
    p : Polynomial S
    r : R
    hp : Membership.mem (Polynomial.mapAlg R S).range p
    ⊢ Membership.mem (Polynomial.mapAlg R S).range (HSMul.hSMul r p)
  -/
  exact Subalgebra.smul_mem (mapAlg R S).range hp r
  /-
    🎉 no goals
  -/


