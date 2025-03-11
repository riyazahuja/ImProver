local notation "q" => Fintype.card K


theorem MvPolynomial.sum_eval_eq_zero (f : MvPolynomial σ K)
    (h : f.totalDegree < (q - 1) * Fintype.card σ) : ∑ x, eval x f = 0 := by
  /-
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    ⊢ Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) f) 0
  -/
  haveI : DecidableEq K := Classical.decEq K
  calc
    ∑ x, eval x f = ∑ x : σ → K, ∑ d ∈ f.support, f.coeff d * ∏ i, x i ^ d i := by
      simp only [eval_eq']
    _ = ∑ d ∈ f.support, ∑ x : σ → K, f.coeff d * ∏ i, x i ^ d i := sum_comm
    _ = 0 := sum_eq_zero ?_
  /-
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this : DecidableEq K
    ⊢ ∀ (x : Finsupp σ Nat), Membership.mem f.support x → Eq (Finset.univ.sum fun  …
  -/
  intro d hd
  /-
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this : DecidableEq K
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    ⊢ Eq (Finset.univ.sum fun x => HMul.hMul (MvPolynomial.coeff d f) (Finset.univ …
  -/
  obtain ⟨i, hi⟩ : ∃ i, d i < q - 1 := f.exists_degree_lt (q - 1) h hd
  calc
    (∑ x : σ → K, f.coeff d * ∏ i, x i ^ d i) = f.coeff d * ∑ x : σ → K, ∏ i, x i ^ d i :=
      (mul_sum ..).symm
    _ = 0 := (mul_eq_zero.mpr ∘ Or.inr) ?_
  calc
    (∑ x : σ → K, ∏ i, x i ^ d i) =
        ∑ x₀ : { j // j ≠ i } → K, ∑ x : { x : σ → K // x ∘ (↑) = x₀ }, ∏ j, (x : σ → K) j ^ d j :=
      (Fintype.sum_fiberwise _ _).symm
    _ = 0 := Fintype.sum_eq_zero _ ?_
  /-
    case intro
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this : DecidableEq K
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    i : σ
    hi : LT.lt (d i) (HSub.hSub (Fintype.card K) 1)
    ⊢ ∀ (a : (Subtype fun j => Ne j i) → K), Eq (Finset.univ.sum fun x => Finset.u …
  -/
  intro x₀
  /-
    case intro
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this : DecidableEq K
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    i : σ
    hi : LT.lt (d i) (HSub.hSub (Fintype.card K) 1)
    x₀ : (Subtype fun j => Ne j i) → K
    ⊢ Eq (Finset.univ.sum fun x => Finset.univ.prod fun j => HPow.hPow (↑x j) (d j …
  -/
  let e : K ≃ { x // x ∘ ((↑) : _ → σ) = x₀ } := (Equiv.subtypeEquivCodomain _).symm
  calc
    (∑ x : { x : σ → K // x ∘ (↑) = x₀ }, ∏ j, (x : σ → K) j ^ d j) =
        ∑ a : K, ∏ j : σ, (e a : σ → K) j ^ d j := (e.sum_comp _).symm
    _ = ∑ a : K, (∏ j, x₀ j ^ d j) * a ^ d i := Fintype.sum_congr _ _ ?_
    _ = (∏ j, x₀ j ^ d j) * ∑ a : K, a ^ d i := by rw [mul_sum]
    _ = 0 := by rw [sum_pow_lt_card_sub_one K _ hi, mul_zero]
  /-
    case intro
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this : DecidableEq K
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    i : σ
    hi : LT.lt (d i) (HSub.hSub (Fintype.card K) 1)
    x₀ : (Subtype fun j => Ne j i) → K
    e : Equiv K (Subtype fun x => Eq (Function.comp x Subtype.val) x₀) := (Equiv.s …
    ⊢ ∀ (a : K), Eq (Finset.univ.prod fun j => HPow.hPow (↑(e a) j) (d j)) (HMul.h …
  -/
  intro a
  /-
    case intro
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this : DecidableEq K
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    i : σ
    hi : LT.lt (d i) (HSub.hSub (Fintype.card K) 1)
    x₀ : (Subtype fun j => Ne j i) → K
    e : Equiv K (Subtype fun x => Eq (Function.comp x Subtype.val) x₀) := (Equiv.s …
    a : K
    ⊢ Eq (Finset.univ.prod fun j => HPow.hPow (↑(e a) j) (d j)) (HMul.hMul (Finset …
  -/
  let e' : { j // j = i } ⊕ { j // j ≠ i } ≃ σ := Equiv.sumCompl _
  letI : Unique { j // j = i } :=
    { default := ⟨i, rfl⟩
      uniq := fun ⟨j, h⟩ => Subtype.val_injective h }
  calc
    (∏ j : σ, (e a : σ → K) j ^ d j) =
        (e a : σ → K) i ^ d i * ∏ j : { j // j ≠ i }, (e a : σ → K) j ^ d j := by
      rw [← e'.prod_comp, Fintype.prod_sum_type, univ_unique, prod_singleton]; rfl
    _ = a ^ d i * ∏ j : { j // j ≠ i }, (e a : σ → K) j ^ d j := by
      rw [Equiv.subtypeEquivCodomain_symm_apply_eq]
    _ = a ^ d i * ∏ j, x₀ j ^ d j := congr_arg _ (Fintype.prod_congr _ _ ?_)
    -- see below
    _ = (∏ j, x₀ j ^ d j) * a ^ d i := mul_comm _ _
  -- the remaining step of the calculation above
  /-
    case intro
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this✝ : DecidableEq K
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    i : σ
    hi : LT.lt (d i) (HSub.hSub (Fintype.card K) 1)
    x₀ : (Subtype fun j => Ne j i) → K
    e : Equiv K (Subtype fun x => Eq (Function.comp x Subtype.val) x₀) := (Equiv.s …
    a : K
    e' : Equiv (Sum (Subtype fun j => Eq j i) (Subtype fun j => Ne j i)) σ := Equi …
    this : Unique (Subtype fun j => Eq j i) := { default := ⟨i, ⋯⟩, uniq := ⋯ }
    ⊢ ∀ (a_1 : Subtype fun j => Ne j i), Eq (HPow.hPow (↑(e a) ↑a_1) (d ↑a_1)) (HP …
  -/
  rintro ⟨j, hj⟩
  /-
    case intro.mk
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this✝ : DecidableEq K
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    i : σ
    hi : LT.lt (d i) (HSub.hSub (Fintype.card K) 1)
    x₀ : (Subtype fun j => Ne j i) → K
    e : Equiv K (Subtype fun x => Eq (Function.comp x Subtype.val) x₀) := (Equiv.s …
    a : K
    e' : Equiv (Sum (Subtype fun j => Eq j i) (Subtype fun j => Ne j i)) σ := Equi …
    this : Unique (Subtype fun j => Eq j i) := { default := ⟨i, ⋯⟩, uniq := ⋯ }
    j : σ
    hj : Ne j i
    ⊢ Eq (HPow.hPow (↑(e a) ↑⟨j, hj⟩) (d ↑⟨j, hj⟩)) (HPow.hPow (x₀ ⟨j, hj⟩) (d ↑⟨j …
  -/
  show (e a : σ → K) j ^ d j = x₀ ⟨j, hj⟩ ^ d j
  /-
    case intro.mk
    K : Type u_1
    σ : Type u_2
    inst✝³ : Fintype K
    inst✝² : Field K
    inst✝¹ : Fintype σ
    inst✝ : DecidableEq σ
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.car …
    this✝ : DecidableEq K
    d : Finsupp σ Nat
    hd : Membership.mem f.support d
    i : σ
    hi : LT.lt (d i) (HSub.hSub (Fintype.card K) 1)
    x₀ : (Subtype fun j => Ne j i) → K
    e : Equiv K (Subtype fun x => Eq (Function.comp x Subtype.val) x₀) := (Equiv.s …
    a : K
    e' : Equiv (Sum (Subtype fun j => Eq j i) (Subtype fun j => Ne j i)) σ := Equi …
    this : Unique (Subtype fun j => Eq j i) := { default := ⟨i, ⋯⟩, uniq := ⋯ }
    j : σ
    hj : Ne j i
    ⊢ Eq (HPow.hPow (↑(e a) j) (d j)) (HPow.hPow (x₀ ⟨j, hj⟩) (d j))
  -/
  rw [Equiv.subtypeEquivCodomain_symm_apply_ne]
  /-
    🎉 no goals
  -/


/-- The **Chevalley–Warning theorem**, finitary version.
Let `(f i)` be a finite family of multivariate polynomials
in finitely many variables (`X s`, `s : σ`) over a finite field of characteristic `p`.
Assume that the sum of the total degrees of the `f i` is less than the cardinality of `σ`.
Then the number of common solutions of the `f i` is divisible by `p`. -/
theorem char_dvd_card_solutions_of_sum_lt {s : Finset ι} {f : ι → MvPolynomial σ K}
    (h : (∑ i ∈ s, (f i).totalDegree) < Fintype.card σ) :
    p ∣ Fintype.card { x : σ → K // ∀ i ∈ s, eval x (f i) = 0 } := by
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => ∀ (i : ι), Membership.mem s i → Eq …
  -/
  have hq : 0 < q - 1 := by rw [← Fintype.card_units, Fintype.card_pos_iff]; exact ⟨1⟩
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => ∀ (i : ι), Membership.mem s i → Eq …
  -/
  let S : Finset (σ → K) := {x | ∀ i ∈ s, eval x (f i) = 0}
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    S : Finset (σ → K) := Finset.filter (fun x => ∀ (i : ι), Membership.mem s i →  …
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => ∀ (i : ι), Membership.mem s i → Eq …
  -/
  have hS (x : σ → K) : x ∈ S ↔ ∀ i ∈ s, eval x (f i) = 0 := by simp [S]
  /- The polynomial `F = ∏ i in s, (1 - (f i)^(q - 1))` has the nice property
    that it takes the value `1` on elements of `{x : σ → K // ∀ i ∈ s, (f i).eval x = 0}`
    while it is `0` outside that locus.
    Hence the sum of its values is equal to the cardinality of
    `{x : σ → K // ∀ i ∈ s, (f i).eval x = 0}` modulo `p`. -/
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    S : Finset (σ → K) := Finset.filter (fun x => ∀ (i : ι), Membership.mem s i →  …
    hS : ∀ (x : σ → K), Iff (Membership.mem S x) (∀ (i : ι), Membership.mem s i →  …
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => ∀ (i : ι), Membership.mem s i → Eq …
  -/
  let F : MvPolynomial σ K := ∏ i ∈ s, (1 - f i ^ (q - 1))
  have hF : ∀ x, eval x F = if x ∈ S then 1 else 0 := by
    intro x
    calc
      eval x F = ∏ i ∈ s, eval x (1 - f i ^ (q - 1)) := eval_prod s _ x
      _ = if x ∈ S then 1 else 0 := ?_
    simp only [(eval x).map_sub, (eval x).map_pow, (eval x).map_one]
    split_ifs with hx
    · apply Finset.prod_eq_one
      intro i hi
      rw [hS] at hx
      rw [hx i hi, zero_pow hq.ne', sub_zero]
    · obtain ⟨i, hi, hx⟩ : ∃ i ∈ s, eval x (f i) ≠ 0 := by
        simpa [hS, not_forall, Classical.not_imp] using hx
      apply Finset.prod_eq_zero hi
      rw [pow_card_sub_one_eq_one (eval x (f i)) hx, sub_self]
  -- In particular, we can now show:
  have key : ∑ x, eval x F = Fintype.card { x : σ → K // ∀ i ∈ s, eval x (f i) = 0 } := by
    rw [Fintype.card_of_subtype S hS, card_eq_sum_ones, Nat.cast_sum, Nat.cast_one, ←
      Fintype.sum_extend_by_zero S, sum_congr rfl fun x _ => hF x]
  -- With these preparations under our belt, we will approach the main goal.
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    S : Finset (σ → K) := Finset.filter (fun x => ∀ (i : ι), Membership.mem s i →  …
    hS : ∀ (x : σ → K), Iff (Membership.mem S x) (∀ (i : ι), Membership.mem s i →  …
    F : MvPolynomial σ K := s.prod fun i => HSub.hSub 1 (HPow.hPow (f i) (HSub.hSu …
    hF : ∀ (x : σ → K), Eq ((MvPolynomial.eval x) F) (ite (Membership.mem S x) 1 0)
    key : Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) F) ↑(Fintype.card (Su …
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => ∀ (i : ι), Membership.mem s i → Eq …
  -/
  show p ∣ Fintype.card { x // ∀ i : ι, i ∈ s → eval x (f i) = 0 }
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    S : Finset (σ → K) := Finset.filter (fun x => ∀ (i : ι), Membership.mem s i →  …
    hS : ∀ (x : σ → K), Iff (Membership.mem S x) (∀ (i : ι), Membership.mem s i →  …
    F : MvPolynomial σ K := s.prod fun i => HSub.hSub 1 (HPow.hPow (f i) (HSub.hSu …
    hF : ∀ (x : σ → K), Eq ((MvPolynomial.eval x) F) (ite (Membership.mem S x) 1 0)
    key : Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) F) ↑(Fintype.card (Su …
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => ∀ (i : ι), Membership.mem s i → Eq …
  -/
  rw [← CharP.cast_eq_zero_iff K, ← key]
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    S : Finset (σ → K) := Finset.filter (fun x => ∀ (i : ι), Membership.mem s i →  …
    hS : ∀ (x : σ → K), Iff (Membership.mem S x) (∀ (i : ι), Membership.mem s i →  …
    F : MvPolynomial σ K := s.prod fun i => HSub.hSub 1 (HPow.hPow (f i) (HSub.hSu …
    hF : ∀ (x : σ → K), Eq ((MvPolynomial.eval x) F) (ite (Membership.mem S x) 1 0)
    key : Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) F) ↑(Fintype.card (Su …
    ⊢ Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) F) 0
  -/
  show (∑ x, eval x F) = 0
  -- We are now ready to apply the main machine, proven before.
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    S : Finset (σ → K) := Finset.filter (fun x => ∀ (i : ι), Membership.mem s i →  …
    hS : ∀ (x : σ → K), Iff (Membership.mem S x) (∀ (i : ι), Membership.mem s i →  …
    F : MvPolynomial σ K := s.prod fun i => HSub.hSub 1 (HPow.hPow (f i) (HSub.hSu …
    hF : ∀ (x : σ → K), Eq ((MvPolynomial.eval x) F) (ite (Membership.mem S x) 1 0)
    key : Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) F) ↑(Fintype.card (Su …
    ⊢ Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) F) 0
  -/
  apply F.sum_eval_eq_zero
  -- It remains to verify the crucial assumption of this machine
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    S : Finset (σ → K) := Finset.filter (fun x => ∀ (i : ι), Membership.mem s i →  …
    hS : ∀ (x : σ → K), Iff (Membership.mem S x) (∀ (i : ι), Membership.mem s i →  …
    F : MvPolynomial σ K := s.prod fun i => HSub.hSub 1 (HPow.hPow (f i) (HSub.hSu …
    hF : ∀ (x : σ → K), Eq ((MvPolynomial.eval x) F) (ite (Membership.mem S x) 1 0)
    key : Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) F) ↑(Fintype.card (Su …
    ⊢ LT.lt F.totalDegree (HMul.hMul (HSub.hSub (Fintype.card K) 1) (Fintype.card  …
  -/
  show F.totalDegree < (q - 1) * Fintype.card σ
  calc
    F.totalDegree ≤ ∑ i ∈ s, (1 - f i ^ (q - 1)).totalDegree := totalDegree_finset_prod s _
    _ ≤ ∑ i ∈ s, (q - 1) * (f i).totalDegree := sum_le_sum fun i _ => ?_
    -- see ↓
    _ = (q - 1) * ∑ i ∈ s, (f i).totalDegree := (mul_sum ..).symm
    _ < (q - 1) * Fintype.card σ := by rwa [mul_lt_mul_left hq]
  -- Now we prove the remaining step from the preceding calculation
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    s : Finset ι
    f : ι → MvPolynomial σ K
    h : LT.lt (s.sum fun i => (f i).totalDegree) (Fintype.card σ)
    hq : LT.lt 0 (HSub.hSub (Fintype.card K) 1)
    S : Finset (σ → K) := Finset.filter (fun x => ∀ (i : ι), Membership.mem s i →  …
    hS : ∀ (x : σ → K), Iff (Membership.mem S x) (∀ (i : ι), Membership.mem s i →  …
    F : MvPolynomial σ K := s.prod fun i => HSub.hSub 1 (HPow.hPow (f i) (HSub.hSu …
    hF : ∀ (x : σ → K), Eq ((MvPolynomial.eval x) F) (ite (Membership.mem S x) 1 0)
    key : Eq (Finset.univ.sum fun x => (MvPolynomial.eval x) F) ↑(Fintype.card (Su …
    i : ι
    x✝ : Membership.mem s i
    ⊢ LE.le (HSub.hSub 1 (HPow.hPow (f i) (HSub.hSub (Fintype.card K) 1))).totalDe …
  -/
  show (1 - f i ^ (q - 1)).totalDegree ≤ (q - 1) * (f i).totalDegree
  calc
    (1 - f i ^ (q - 1)).totalDegree ≤
        max (1 : MvPolynomial σ K).totalDegree (f i ^ (q - 1)).totalDegree := totalDegree_sub _ _
    _ ≤ (f i ^ (q - 1)).totalDegree := by simp
    _ ≤ (q - 1) * (f i).totalDegree := totalDegree_pow _ _


/-- The **Chevalley–Warning theorem**, `Fintype` version.
Let `(f i)` be a finite family of multivariate polynomials
in finitely many variables (`X s`, `s : σ`) over a finite field of characteristic `p`.
Assume that the sum of the total degrees of the `f i` is less than the cardinality of `σ`.
Then the number of common solutions of the `f i` is divisible by `p`. -/
theorem char_dvd_card_solutions_of_fintype_sum_lt [Fintype ι] {f : ι → MvPolynomial σ K}
    (h : (∑ i, (f i).totalDegree) < Fintype.card σ) :
    p ∣ Fintype.card { x : σ → K // ∀ i, eval x (f i) = 0 } := by
  /-
    K : Type u_1
    σ : Type u_2
    ι : Type u_3
    inst✝⁶ : Fintype K
    inst✝⁵ : Field K
    inst✝⁴ : Fintype σ
    inst✝³ : DecidableEq σ
    inst✝² : DecidableEq K
    p : Nat
    inst✝¹ : CharP K p
    inst✝ : Fintype ι
    f : ι → MvPolynomial σ K
    h : LT.lt (Finset.univ.sum fun i => (f i).totalDegree) (Fintype.card σ)
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => ∀ (i : ι), Eq ((MvPolynomial.eval  …
  -/
  simpa using char_dvd_card_solutions_of_sum_lt p h
  /-
    🎉 no goals
  -/


/-- The **Chevalley–Warning theorem**, unary version.
Let `f` be a multivariate polynomial in finitely many variables (`X s`, `s : σ`)
over a finite field of characteristic `p`.
Assume that the total degree of `f` is less than the cardinality of `σ`.
Then the number of solutions of `f` is divisible by `p`.
See `char_dvd_card_solutions_of_sum_lt` for a version that takes a family of polynomials `f i`. -/
theorem char_dvd_card_solutions {f : MvPolynomial σ K} (h : f.totalDegree < Fintype.card σ) :
    p ∣ Fintype.card { x : σ → K // eval x f = 0 } := by
  /-
    K : Type u_1
    σ : Type u_2
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (Fintype.card σ)
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => Eq ((MvPolynomial.eval x) f) 0))
  -/
  let F : Unit → MvPolynomial σ K := fun _ => f
  /-
    K : Type u_1
    σ : Type u_2
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (Fintype.card σ)
    F : Unit → MvPolynomial σ K := fun x => f
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => Eq ((MvPolynomial.eval x) f) 0))
  -/
  have : (∑ i : Unit, (F i).totalDegree) < Fintype.card σ := h
  -- Porting note: was
  -- `simpa only [F, Fintype.univ_punit, forall_eq, mem_singleton] using`
  -- `  char_dvd_card_solutions_of_sum_lt p this`
  /-
    K : Type u_1
    σ : Type u_2
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (Fintype.card σ)
    F : Unit → MvPolynomial σ K := fun x => f
    this : LT.lt (Finset.univ.sum fun i => (F i).totalDegree) (Fintype.card σ)
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => Eq ((MvPolynomial.eval x) f) 0))
  -/
  convert char_dvd_card_solutions_of_sum_lt p this
  /-
    case h.e'_4.h.h.e'_2.h.a
    K : Type u_1
    σ : Type u_2
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    f : MvPolynomial σ K
    h : LT.lt f.totalDegree (Fintype.card σ)
    F : Unit → MvPolynomial σ K := fun x => f
    this : LT.lt (Finset.univ.sum fun i => (F i).totalDegree) (Fintype.card σ)
    x✝ : σ → K
    ⊢ Iff (Eq ((MvPolynomial.eval x✝) f) 0) (∀ (i : Unit), Membership.mem Finset.u …
  -/
  aesop
  /-
    🎉 no goals
  -/


/-- The **Chevalley–Warning theorem**, binary version.
Let `f₁`, `f₂` be two multivariate polynomials in finitely many variables (`X s`, `s : σ`) over a
finite field of characteristic `p`.
Assume that the sum of the total degrees of `f₁` and `f₂` is less than the cardinality of `σ`.
Then the number of common solutions of the `f₁` and `f₂` is divisible by `p`. -/
theorem char_dvd_card_solutions_of_add_lt {f₁ f₂ : MvPolynomial σ K}
    (h : f₁.totalDegree + f₂.totalDegree < Fintype.card σ) :
    p ∣ Fintype.card { x : σ → K // eval x f₁ = 0 ∧ eval x f₂ = 0 } := by
  /-
    K : Type u_1
    σ : Type u_2
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    f₁ f₂ : MvPolynomial σ K
    h : LT.lt (HAdd.hAdd f₁.totalDegree f₂.totalDegree) (Fintype.card σ)
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => And (Eq ((MvPolynomial.eval x) f₁) …
  -/
  let F : Bool → MvPolynomial σ K := fun b => cond b f₂ f₁
  /-
    K : Type u_1
    σ : Type u_2
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    f₁ f₂ : MvPolynomial σ K
    h : LT.lt (HAdd.hAdd f₁.totalDegree f₂.totalDegree) (Fintype.card σ)
    F : Bool → MvPolynomial σ K := fun b => cond b f₂ f₁
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => And (Eq ((MvPolynomial.eval x) f₁) …
  -/
  have : (∑ b : Bool, (F b).totalDegree) < Fintype.card σ := (add_comm _ _).trans_lt h
  /-
    K : Type u_1
    σ : Type u_2
    inst✝⁵ : Fintype K
    inst✝⁴ : Field K
    inst✝³ : Fintype σ
    inst✝² : DecidableEq σ
    inst✝¹ : DecidableEq K
    p : Nat
    inst✝ : CharP K p
    f₁ f₂ : MvPolynomial σ K
    h : LT.lt (HAdd.hAdd f₁.totalDegree f₂.totalDegree) (Fintype.card σ)
    F : Bool → MvPolynomial σ K := fun b => cond b f₂ f₁
    this : LT.lt (Finset.univ.sum fun b => (F b).totalDegree) (Fintype.card σ)
    ⊢ Dvd.dvd p (Fintype.card (Subtype fun x => And (Eq ((MvPolynomial.eval x) f₁) …
  -/
  simpa only [Bool.forall_bool] using char_dvd_card_solutions_of_fintype_sum_lt p this
  /-
    🎉 no goals
  -/


