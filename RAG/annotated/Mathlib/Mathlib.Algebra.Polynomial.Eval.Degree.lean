theorem eval₂_eq_sum_range :
    p.eval₂ f x = ∑ i ∈ Finset.range (p.natDegree + 1), f (p.coeff i) * x ^ i :=
  _root_.trans (congr_arg _ p.as_sum_range)
                                                              /-
                                                                R : Type u
                                                                S : Type v
                                                                inst✝¹ : Semiring R
                                                                p : Polynomial R
                                                                inst✝ : Semiring S
                                                                f : RingHom R S
                                                                x : S
                                                                ⊢ Eq (fun i => Polynomial.eval₂ f x ((Polynomial.monomial i) (p.coeff i))) fun …
                                                              -/
    (_root_.trans (eval₂_finset_sum f _ _ x) (congr_arg _ (by simp)))
                                                              /-
                                                                🎉 no goals
                                                              -/


theorem eval₂_eq_sum_range' (f : R →+* S) {p : R[X]} {n : ℕ} (hn : p.natDegree < n) (x : S) :
    eval₂ f x p = ∑ i ∈ Finset.range n, f (p.coeff i) * x ^ i := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    n : Nat
    hn : LT.lt p.natDegree n
    x : S
    ⊢ Eq (Polynomial.eval₂ f x p) ((Finset.range n).sum fun i => HMul.hMul (f (p.c …
  -/
  rw [eval₂_eq_sum, p.sum_over_range' _ _ hn]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    n : Nat
    hn : LT.lt p.natDegree n
    x : S
    ⊢ ∀ (n : Nat), Eq (HMul.hMul (f 0) (HPow.hPow x n)) 0
  -/
  intro i
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    n : Nat
    hn : LT.lt p.natDegree n
    x : S
    i : Nat
    ⊢ Eq (HMul.hMul (f 0) (HPow.hPow x i)) 0
  -/
  rw [f.map_zero, zero_mul]
  /-
    🎉 no goals
  -/


theorem eval_eq_sum_range {p : R[X]} (x : R) :
    p.eval x = ∑ i ∈ Finset.range (p.natDegree + 1), p.coeff i * x ^ i := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    x : R
    ⊢ Eq (Polynomial.eval x p) ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fun i …
  -/
  rw [eval_eq_sum, sum_over_range]; simp
                                    /-
                                      🎉 no goals
                                    -/


theorem eval_eq_sum_range' {p : R[X]} {n : ℕ} (hn : p.natDegree < n) (x : R) :
    p.eval x = ∑ i ∈ Finset.range n, p.coeff i * x ^ i := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    n : Nat
    hn : LT.lt p.natDegree n
    x : R
    ⊢ Eq (Polynomial.eval x p) ((Finset.range n).sum fun i => HMul.hMul (p.coeff i …
  -/
  rw [eval_eq_sum, p.sum_over_range' _ _ hn]; simp
                                              /-
                                                🎉 no goals
                                              -/


/-- A reformulation of the expansion of (1 + y)^d:
$$(d + 1) (1 + y)^d - (d + 1)y^d = \sum_{i = 0}^d {d + 1 \choose i} \cdot i \cdot y^{i - 1}.$$
-/
theorem eval_monomial_one_add_sub [CommRing S] (d : ℕ) (y : S) :
    eval (1 + y) (monomial d (d + 1 : S)) - eval y (monomial d (d + 1 : S)) =
      ∑ x_1 ∈ range (d + 1), ↑((d + 1).choose x_1) * (↑x_1 * y ^ (x_1 - 1)) := by
  /-
    S : Type v
    inst✝ : CommRing S
    d : Nat
    y : S
    ⊢ Eq (HSub.hSub (Polynomial.eval (HAdd.hAdd 1 y) ((Polynomial.monomial d) (HAd …
  -/
  have cast_succ : (d + 1 : S) = ((d.succ : ℕ) : S) := by simp only [Nat.cast_succ]
  /-
    S : Type v
    inst✝ : CommRing S
    d : Nat
    y : S
    cast_succ : Eq (HAdd.hAdd (↑d) 1) ↑d.succ
    ⊢ Eq (HSub.hSub (Polynomial.eval (HAdd.hAdd 1 y) ((Polynomial.monomial d) (HAd …
  -/
  rw [cast_succ, eval_monomial, eval_monomial, add_comm, add_pow]
  -- Porting note: `apply_congr` hadn't been ported yet, so `congr` & `ext` is used.
  conv_lhs =>
    congr
    · congr
      · skip
      · congr
        · skip
        · ext
          rw [one_pow, mul_one, mul_comm]
  rw [sum_range_succ, mul_add, Nat.choose_self, Nat.cast_one, one_mul, add_sub_cancel_right,
    mul_sum, sum_range_succ', Nat.cast_zero, zero_mul, mul_zero, add_zero]
  /-
    S : Type v
    inst✝ : CommRing S
    d : Nat
    y : S
    cast_succ : Eq (HAdd.hAdd (↑d) 1) ↑d.succ
    ⊢ Eq ((Finset.range d).sum fun i => HMul.hMul (↑d.succ) (HMul.hMul (↑(d.choose …
  -/
  refine sum_congr rfl fun y _hy => ?_
  rw [← mul_assoc, ← mul_assoc, ← Nat.cast_mul, Nat.succ_mul_choose_eq, Nat.cast_mul,
    Nat.add_sub_cancel]


theorem coeff_comp_degree_mul_degree (hqd0 : natDegree q ≠ 0) :
    coeff (p.comp q) (natDegree p * natDegree q) =
    leadingCoeff p * leadingCoeff q ^ natDegree p := by
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hqd0 : Ne q.natDegree 0
    ⊢ Eq ((p.comp q).coeff (HMul.hMul p.natDegree q.natDegree)) (HMul.hMul p.leadi …
  -/
  rw [comp, eval₂_def, coeff_sum]
  -- Porting note: `convert` → `refine`
  /-
    R : Type u
    inst✝ : Semiring R
    p q : Polynomial R
    hqd0 : Ne q.natDegree 0
    ⊢ Eq (p.sum fun a b => (HMul.hMul (Polynomial.C b) (HPow.hPow q a)).coeff (HMu …
  -/
  refine Eq.trans (Finset.sum_eq_single p.natDegree ?h₀ ?h₁) ?h₂
  case h₂ =>
    simp only [coeff_natDegree, coeff_C_mul, coeff_pow_mul_natDegree]
  case h₀ =>
    intro b hbs hbp
    refine coeff_eq_zero_of_natDegree_lt (natDegree_mul_le.trans_lt ?_)
    rw [natDegree_C, zero_add]
    refine natDegree_pow_le.trans_lt ((mul_lt_mul_right (pos_iff_ne_zero.mpr hqd0)).mpr ?_)
    exact lt_of_le_of_ne (le_natDegree_of_mem_supp _ hbs) hbp
  case h₁ =>
    simp +contextual


@[simp] lemma comp_C_mul_X_coeff {r : R} {n : ℕ} :
    (p.comp <| C r * X).coeff n = p.coeff n * r ^ n := by
  simp_rw [comp, eval₂_eq_sum_range, (commute_X _).symm.mul_pow,
    ← C_pow, finset_sum_coeff, coeff_C_mul, coeff_X_pow]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    r : R
    n : Nat
    ⊢ Eq ((Finset.range (HAdd.hAdd p.natDegree 1)).sum fun x => HMul.hMul (p.coeff …
  -/
  rw [Finset.sum_eq_single n _ fun h ↦ ?_, if_pos rfl, mul_one]
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      r : R
      n : Nat
      ⊢ ∀ (b : Nat), Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) b → Ne  …
    -/
  · intro b _ h; simp_rw [if_neg h.symm, mul_zero]
                 /-
                   🎉 no goals
                 -/
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      r : R
      n : Nat
      h : Not (Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) n)
      ⊢ Eq (HMul.hMul (p.coeff n) (HMul.hMul (HPow.hPow r n) (ite (Eq n n) 1 0))) 0
    -/
  · rw [coeff_eq_zero_of_natDegree_lt, zero_mul]
    /-
      R : Type u
      inst✝ : Semiring R
      p : Polynomial R
      r : R
      n : Nat
      h : Not (Membership.mem (Finset.range (HAdd.hAdd p.natDegree 1)) n)
      ⊢ LT.lt p.natDegree n
    -/
    rwa [Finset.mem_range_succ_iff, not_le] at h
    /-
      🎉 no goals
    -/


lemma comp_C_mul_X_eq_zero_iff {r : R} (hr : r ∈ nonZeroDivisors R) :
    p.comp (C r * X) = 0 ↔ p = 0 := by
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    r : R
    hr : Membership.mem (nonZeroDivisors R) r
    ⊢ Iff (Eq (p.comp (HMul.hMul (Polynomial.C r) Polynomial.X)) 0) (Eq p 0)
  -/
  simp_rw [ext_iff]
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    r : R
    hr : Membership.mem (nonZeroDivisors R) r
    ⊢ Iff (∀ (n : Nat), Eq ((p.comp (HMul.hMul (Polynomial.C r) Polynomial.X)).coe …
  -/
  refine forall_congr' fun n ↦ ?_
  /-
    R : Type u
    inst✝ : Semiring R
    p : Polynomial R
    r : R
    hr : Membership.mem (nonZeroDivisors R) r
    n : Nat
    ⊢ Iff (Eq ((p.comp (HMul.hMul (Polynomial.C r) Polynomial.X)).coeff n) (Polyno …
  -/
  rw [comp_C_mul_X_coeff, coeff_zero, mul_right_mem_nonZeroDivisors_eq_zero_iff (pow_mem hr _)]
  /-
    🎉 no goals
  -/


variable (f) in
/-- If `R` and `S` are isomorphic, then so are their polynomial rings. -/
@[simps!]
def mapEquiv (e : R ≃+* S) : R[X] ≃+* S[X] :=
                                                                                    /-
                                                                                      R : Type u
                                                                                      S : Type v
                                                                                      T : Type w
                                                                                      ι : Type y
                                                                                      a b : R
                                                                                      m n : Nat
                                                                                      inst✝¹ : Semiring R
                                                                                      p✝ q r : Polynomial R
                                                                                      inst✝ : Semiring S
                                                                                      f : RingHom R S
                                                                                      p : Polynomial R
                                                                                      e : RingEquiv R S
                                                                                      ⊢ Eq ((↑(Polynomial.mapRingHom ↑e.symm)).comp ↑(Polynomial.mapRingHom ↑e)) (Ri …
                                                                                    -/
  RingEquiv.ofHomInv (mapRingHom (e : R →+* S)) (mapRingHom (e.symm : S →+* R)) (by ext; simp)
                                                                                         /-
                                                                                           🎉 no goals
                                                                                         -/
        /-
          R : Type u
          S : Type v
          T : Type w
          ι : Type y
          a b : R
          m n : Nat
          inst✝¹ : Semiring R
          p✝ q r : Polynomial R
          inst✝ : Semiring S
          f : RingHom R S
          p : Polynomial R
          e : RingEquiv R S
          ⊢ Eq ((↑(Polynomial.mapRingHom ↑e)).comp ↑(Polynomial.mapRingHom ↑e.symm)) (Ri …
        -/
    (by ext; simp)
             /-
               🎉 no goals
             -/


theorem map_monic_eq_zero_iff (hp : p.Monic) : p.map f = 0 ↔ ∀ x, f x = 0 :=
  ⟨fun hfp x =>
    calc
                                         /-
                                           R : Type u
                                           S : Type v
                                           inst✝¹ : Semiring R
                                           inst✝ : Semiring S
                                           f : RingHom R S
                                           p : Polynomial R
                                           hp : p.Monic
                                           hfp : Eq (Polynomial.map f p) 0
                                           x : R
                                           ⊢ Eq (f x) (HMul.hMul (f x) (f p.leadingCoeff))
                                         -/
      f x = f x * f p.leadingCoeff := by simp only [mul_one, hp.leadingCoeff, f.map_one]
                                         /-
                                           🎉 no goals
                                         -/
      _ = f x * (p.map f).coeff p.natDegree := congr_arg _ (coeff_map _ _).symm
                  /-
                    R : Type u
                    S : Type v
                    inst✝¹ : Semiring R
                    inst✝ : Semiring S
                    f : RingHom R S
                    p : Polynomial R
                    hp : p.Monic
                    hfp : Eq (Polynomial.map f p) 0
                    x : R
                    ⊢ Eq (HMul.hMul (f x) ((Polynomial.map f p).coeff p.natDegree)) 0
                  -/
      _ = 0 := by simp only [hfp, mul_zero, coeff_zero]
                  /-
                    🎉 no goals
                  -/
      ,
                             /-
                               R : Type u
                               S : Type v
                               inst✝¹ : Semiring R
                               inst✝ : Semiring S
                               f : RingHom R S
                               p : Polynomial R
                               hp : p.Monic
                               h : ∀ (x : R), Eq (f x) 0
                               n : Nat
                               ⊢ Eq ((Polynomial.map f p).coeff n) (Polynomial.coeff 0 n)
                             -/
    fun h => ext fun n => by simp only [h, coeff_map, coeff_zero]⟩
                             /-
                               🎉 no goals
                             -/


theorem map_monic_ne_zero (hp : p.Monic) [Nontrivial S] : p.map f ≠ 0 := fun h =>
  f.map_one_ne_zero ((map_monic_eq_zero_iff hp).mp h _)


lemma degree_map_le : degree (p.map f) ≤ degree p := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    ⊢ LE.le (Polynomial.map f p).degree p.degree
  -/
  refine (degree_le_iff_coeff_zero _ _).2 fun m hm => ?_
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    m : Nat
    hm : LT.lt p.degree ↑m
    ⊢ Eq ((Polynomial.map f p).coeff m) 0
  -/
  rw [degree_lt_iff_coeff_zero] at hm
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    m : Nat
    hm : ∀ (m_1 : Nat), LE.le m m_1 → Eq (p.coeff m_1) 0
    ⊢ Eq ((Polynomial.map f p).coeff m) 0
  -/
  simp [hm m le_rfl]
  /-
    🎉 no goals
  -/


lemma natDegree_map_le : natDegree (p.map f) ≤ natDegree p := natDegree_le_natDegree degree_map_le


lemma degree_map_lt (hp : f p.leadingCoeff = 0) (hp₀ : p ≠ 0) : (p.map f).degree < p.degree := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    hp : Eq (f p.leadingCoeff) 0
    hp₀ : Ne p 0
    ⊢ LT.lt (Polynomial.map f p).degree p.degree
  -/
  refine degree_map_le.lt_of_ne fun hpq ↦ hp₀ ?_
  rw [leadingCoeff, ← coeff_map, ← natDegree_eq_natDegree hpq, ← leadingCoeff, leadingCoeff_eq_zero]
    at hp
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    hp : Eq (Polynomial.map f p) 0
    hp₀ : Ne p 0
    hpq : Eq (Polynomial.map f p).degree p.degree
    ⊢ Eq p 0
  -/
  rw [← degree_eq_bot, ← hpq, hp, degree_zero]
  /-
    🎉 no goals
  -/


lemma natDegree_map_lt (hp : f p.leadingCoeff = 0) (hp₀ : map f p ≠ 0) :
    (p.map f).natDegree < p.natDegree :=
                                                       /-
                                                         R : Type u
                                                         S : Type v
                                                         inst✝¹ : Semiring R
                                                         inst✝ : Semiring S
                                                         f : RingHom R S
                                                         p : Polynomial R
                                                         hp : Eq (f p.leadingCoeff) 0
                                                         hp₀ : Ne (Polynomial.map f p) 0
                                                         ⊢ Ne p 0
                                                       -/
  natDegree_lt_natDegree hp₀ <| degree_map_lt hp <| by rintro rfl; simp at hp₀
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


/-- Variant of `natDegree_map_lt` that assumes `0 < natDegree p` instead of `map f p ≠ 0`. -/
lemma natDegree_map_lt' (hp : f p.leadingCoeff = 0) (hp₀ : 0 < natDegree p) :
    (p.map f).natDegree < p.natDegree := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    f : RingHom R S
    p : Polynomial R
    hp : Eq (f p.leadingCoeff) 0
    hp₀ : LT.lt 0 p.natDegree
    ⊢ LT.lt (Polynomial.map f p).natDegree p.natDegree
  -/
  by_cases H : map f p = 0
    /-
      case pos
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial R
      hp : Eq (f p.leadingCoeff) 0
      hp₀ : LT.lt 0 p.natDegree
      H : Eq (Polynomial.map f p) 0
      ⊢ LT.lt (Polynomial.map f p).natDegree p.natDegree
    -/
  · rwa [H, natDegree_zero]
    /-
      🎉 no goals
    -/
    /-
      case neg
      R : Type u
      S : Type v
      inst✝¹ : Semiring R
      inst✝ : Semiring S
      f : RingHom R S
      p : Polynomial R
      hp : Eq (f p.leadingCoeff) 0
      hp₀ : LT.lt 0 p.natDegree
      H : Not (Eq (Polynomial.map f p) 0)
      ⊢ LT.lt (Polynomial.map f p).natDegree p.natDegree
    -/
  · exact natDegree_map_lt hp H
    /-
      🎉 no goals
    -/


theorem degree_map_eq_of_leadingCoeff_ne_zero (f : R →+* S) (hf : f (leadingCoeff p) ≠ 0) :
    degree (p.map f) = degree p := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    hf : Ne (f p.leadingCoeff) 0
    ⊢ Eq (Polynomial.map f p).degree p.degree
  -/
  refine degree_map_le.antisymm ?_
  have hp0 : p ≠ 0 :=
    leadingCoeff_ne_zero.mp fun hp0 => hf (_root_.trans (congr_arg _ hp0) f.map_zero)
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    hf : Ne (f p.leadingCoeff) 0
    hp0 : Ne p 0
    ⊢ LE.le p.degree (Polynomial.map f p).degree
  -/
  rw [degree_eq_natDegree hp0]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    hf : Ne (f p.leadingCoeff) 0
    hp0 : Ne p 0
    ⊢ LE.le (↑p.natDegree) (Polynomial.map f p).degree
  -/
  refine le_degree_of_ne_zero ?_
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    hf : Ne (f p.leadingCoeff) 0
    hp0 : Ne p 0
    ⊢ Ne ((Polynomial.map f p).coeff p.natDegree) 0
  -/
  rw [coeff_map]
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    hf : Ne (f p.leadingCoeff) 0
    hp0 : Ne p 0
    ⊢ Ne (f (p.coeff p.natDegree)) 0
  -/
  exact hf
  /-
    🎉 no goals
  -/


theorem natDegree_map_of_leadingCoeff_ne_zero (f : R →+* S) (hf : f (leadingCoeff p) ≠ 0) :
    natDegree (p.map f) = natDegree p :=
  natDegree_eq_of_degree_eq (degree_map_eq_of_leadingCoeff_ne_zero f hf)


theorem leadingCoeff_map_of_leadingCoeff_ne_zero (f : R →+* S) (hf : f (leadingCoeff p) ≠ 0) :
    leadingCoeff (p.map f) = f (leadingCoeff p) := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    hf : Ne (f p.leadingCoeff) 0
    ⊢ Eq (Polynomial.map f p).leadingCoeff (f p.leadingCoeff)
  -/
  unfold leadingCoeff
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    inst✝ : Semiring S
    p : Polynomial R
    f : RingHom R S
    hf : Ne (f p.leadingCoeff) 0
    ⊢ Eq ((Polynomial.map f p).coeff (Polynomial.map f p).natDegree) (f (p.coeff p …
  -/
  rw [coeff_map, natDegree_map_of_leadingCoeff_ne_zero f hf]
  /-
    🎉 no goals
  -/


theorem eval₂_comp {x : S} : eval₂ f x (p.comp q) = eval₂ f (eval₂ f x q) p := by
  /-
    R : Type u
    S : Type v
    inst✝¹ : Semiring R
    p q : Polynomial R
    inst✝ : CommSemiring S
    f : RingHom R S
    x : S
    ⊢ Eq (Polynomial.eval₂ f x (p.comp q)) (Polynomial.eval₂ f (Polynomial.eval₂ f …
  -/
  rw [comp, p.as_sum_range]; simp [eval₂_finset_sum, eval₂_pow]
                             /-
                               🎉 no goals
                             -/


@[simp]
theorem iterate_comp_eval₂ (k : ℕ) (t : S) :
    eval₂ f t (p.comp^[k] q) = (fun x => eval₂ f x p)^[k] (eval₂ f t q) := by
  induction k with
  | zero => simp
  | succ k IH => rw [Function.iterate_succ_apply', Function.iterate_succ_apply', eval₂_comp, IH]


@[simp]
theorem iterate_comp_eval :
    ∀ (k : ℕ) (t : R), (p.comp^[k] q).eval t = (fun x => p.eval x)^[k] (q.eval t) :=
  iterate_comp_eval₂ _


lemma isUnit_of_isUnit_leadingCoeff_of_isUnit_map (hf : IsUnit f.leadingCoeff)
    (H : IsUnit (map φ f)) : IsUnit f := by
  /-
    R : Type u
    S : Type v
    inst✝² : Semiring R
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hf : IsUnit f.leadingCoeff
    H : IsUnit (Polynomial.map φ f)
    ⊢ IsUnit f
  -/
  have dz := degree_eq_zero_of_isUnit H
  /-
    R : Type u
    S : Type v
    inst✝² : Semiring R
    inst✝¹ : CommRing S
    inst✝ : IsDomain S
    φ : RingHom R S
    f : Polynomial R
    hf : IsUnit f.leadingCoeff
    H : IsUnit (Polynomial.map φ f)
    dz : Eq (Polynomial.map φ f).degree 0
    ⊢ IsUnit f
  -/
  rw [degree_map_eq_of_leadingCoeff_ne_zero] at dz
    /-
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq f.degree 0
      ⊢ IsUnit f
    -/
  · rw [eq_C_of_degree_eq_zero dz]
    /-
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq f.degree 0
      ⊢ IsUnit (Polynomial.C (f.coeff 0))
    -/
    refine IsUnit.map C ?_
    /-
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq f.degree 0
      ⊢ IsUnit (f.coeff 0)
    -/
    convert hf
    /-
      case h.e'_3
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq f.degree 0
      ⊢ Eq (f.coeff 0) f.leadingCoeff
    -/
    change coeff f 0 = coeff f (natDegree f)
    /-
      case h.e'_3
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq f.degree 0
      ⊢ Eq (f.coeff 0) (f.coeff f.natDegree)
    -/
    rw [(degree_eq_iff_natDegree_eq _).1 dz]
      /-
        case h.e'_3
        R : Type u
        S : Type v
        inst✝² : Semiring R
        inst✝¹ : CommRing S
        inst✝ : IsDomain S
        φ : RingHom R S
        f : Polynomial R
        hf : IsUnit f.leadingCoeff
        H : IsUnit (Polynomial.map φ f)
        dz : Eq f.degree 0
        ⊢ Eq (f.coeff 0) (f.coeff Zero.zero)
      -/
    · rfl
      /-
        🎉 no goals
      -/
    /-
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq f.degree 0
      ⊢ Ne f 0
    -/
    rintro rfl
    /-
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      hf : IsUnit (Polynomial.leadingCoeff 0)
      H : IsUnit (Polynomial.map φ 0)
      dz : Eq (Polynomial.degree 0) 0
      ⊢ False
    -/
    simp at H
    /-
      🎉 no goals
    -/
    /-
      case hf
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq (Polynomial.map φ f).degree 0
      ⊢ Ne (φ f.leadingCoeff) 0
    -/
  · intro h
    /-
      case hf
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq (Polynomial.map φ f).degree 0
      h : Eq (φ f.leadingCoeff) 0
      ⊢ False
    -/
    have u : IsUnit (φ f.leadingCoeff) := IsUnit.map φ hf
    /-
      case hf
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq (Polynomial.map φ f).degree 0
      h : Eq (φ f.leadingCoeff) 0
      u : IsUnit (φ f.leadingCoeff)
      ⊢ False
    -/
    rw [h] at u
    /-
      case hf
      R : Type u
      S : Type v
      inst✝² : Semiring R
      inst✝¹ : CommRing S
      inst✝ : IsDomain S
      φ : RingHom R S
      f : Polynomial R
      hf : IsUnit f.leadingCoeff
      H : IsUnit (Polynomial.map φ f)
      dz : Eq (Polynomial.map φ f).degree 0
      h : Eq (φ f.leadingCoeff) 0
      u : IsUnit 0
      ⊢ False
    -/
    simp at u
    /-
      🎉 no goals
    -/


