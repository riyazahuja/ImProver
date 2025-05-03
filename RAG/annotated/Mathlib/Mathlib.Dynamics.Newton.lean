/-- Given a single-variable polynomial `P` with derivative `P'`, this is the map:
`x ↦ x - P(x) / P'(x)`. When `P'(x)` is not a unit we use a junk-value pattern and send `x ↦ x`. -/
def newtonMap (x : S) : S :=
  x - (Ring.inverse <| aeval x (derivative P)) * aeval x P


theorem newtonMap_apply :
    P.newtonMap x = x - (Ring.inverse <| aeval x (derivative P)) * (aeval x P) :=
  rfl


theorem newtonMap_apply_of_isUnit (h : IsUnit <| aeval x (derivative P)) :
    P.newtonMap x = x - h.unit⁻¹ * aeval x P := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    x : S
    h : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
    ⊢ Eq (P.newtonMap x) (HSub.hSub x (HMul.hMul (↑(Inv.inv h.unit)) ((Polynomial. …
  -/
  simp [newtonMap_apply, Ring.inverse, h]
  /-
    🎉 no goals
  -/


theorem newtonMap_apply_of_not_isUnit (h : ¬ (IsUnit <| aeval x (derivative P))) :
    P.newtonMap x = x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    x : S
    h : Not (IsUnit ((Polynomial.aeval x) (Polynomial.derivative P)))
    ⊢ Eq (P.newtonMap x) x
  -/
  simp [newtonMap_apply, Ring.inverse, h]
  /-
    🎉 no goals
  -/


theorem isNilpotent_iterate_newtonMap_sub_of_isNilpotent (h : IsNilpotent <| aeval x P) (n : ℕ) :
    IsNilpotent <| P.newtonMap^[n] x - x := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [iterate_succ', comp_apply, newtonMap_apply, sub_right_comm]
    refine (Commute.all _ _).isNilpotent_sub ih <| (Commute.all _ _).isNilpotent_mul_right ?_
    simpa using Commute.isNilpotent_add (Commute.all _ _)
      (isNilpotent_aeval_sub_of_isNilpotent_sub P ih) h


theorem isFixedPt_newtonMap_of_aeval_eq_zero (h : aeval x P = 0) :
    IsFixedPt P.newtonMap x := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    x : S
    h : Eq ((Polynomial.aeval x) P) 0
    ⊢ Function.IsFixedPt P.newtonMap x
  -/
  rw [IsFixedPt, newtonMap_apply, h, mul_zero, sub_zero]
  /-
    🎉 no goals
  -/


theorem isFixedPt_newtonMap_of_isUnit_iff (h : IsUnit <| aeval x (derivative P)) :
    IsFixedPt P.newtonMap x ↔ aeval x P = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    x : S
    h : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
    ⊢ Iff (Function.IsFixedPt P.newtonMap x) (Eq ((Polynomial.aeval x) P) 0)
  -/
  rw [IsFixedPt, newtonMap_apply, sub_eq_self, Ring.inverse_mul_eq_iff_eq_mul _ _ _ h, mul_zero]
  /-
    🎉 no goals
  -/


/-- This is really an auxiliary result, en route to
`Polynomial.existsUnique_nilpotent_sub_and_aeval_eq_zero`. -/
theorem aeval_pow_two_pow_dvd_aeval_iterate_newtonMap
    (h : IsNilpotent (aeval x P)) (h' : IsUnit (aeval x <| derivative P)) (n : ℕ) :
    (aeval x P) ^ (2 ^ n) ∣ aeval (P.newtonMap^[n] x) P := by
  induction n with
  | zero => simp
  | succ n ih =>
    have ⟨d, hd⟩ := binomExpansion (P.map (algebraMap R S)) (P.newtonMap^[n] x)
      (-Ring.inverse (aeval (P.newtonMap^[n] x) <| derivative P) * aeval (P.newtonMap^[n] x) P)
    rw [eval_map_algebraMap, eval_map_algebraMap] at hd
    rw [iterate_succ', comp_apply, newtonMap_apply, sub_eq_add_neg, neg_mul_eq_neg_mul, hd]
    refine dvd_add ?_ (dvd_mul_of_dvd_right ?_ _)
    · convert dvd_zero _
      have : IsUnit (aeval (P.newtonMap^[n] x) <| derivative P) :=
        isUnit_aeval_of_isUnit_aeval_of_isNilpotent_sub h' <|
        isNilpotent_iterate_newtonMap_sub_of_isNilpotent h n
      rw [derivative_map, eval_map_algebraMap, ← mul_assoc, mul_neg, Ring.mul_inverse_cancel _ this,
        neg_mul, one_mul, add_neg_cancel]
    · rw [neg_mul, even_two.neg_pow, mul_pow, pow_succ, pow_mul]
      exact dvd_mul_of_dvd_right (pow_dvd_pow_of_dvd ih 2) _


/-- If `x` is almost a root of `P` in the sense that `P(x)` is nilpotent (and `P'(x)` is a
unit) then we may write `x` as a sum `x = n + r` where `n` is nilpotent and `r` is a root of `P`.
Moreover, `n` and `r` are unique.

This can be used to prove the Jordan-Chevalley decomposition of linear endomorphims. -/
theorem existsUnique_nilpotent_sub_and_aeval_eq_zero
    (h : IsNilpotent (aeval x P)) (h' : IsUnit (aeval x <| derivative P)) :
    ∃! r, IsNilpotent (x - r) ∧ aeval r P = 0 := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    x : S
    h : IsNilpotent ((Polynomial.aeval x) P)
    h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
    ⊢ ExistsUnique fun r => And (IsNilpotent (HSub.hSub x r)) (Eq ((Polynomial.aev …
  -/
  simp_rw [(neg_sub _ x).symm, isNilpotent_neg_iff]
  /-
    R : Type u_1
    S : Type u_2
    inst✝² : CommRing R
    inst✝¹ : CommRing S
    inst✝ : Algebra R S
    P : Polynomial R
    x : S
    h : IsNilpotent ((Polynomial.aeval x) P)
    h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
    ⊢ ExistsUnique fun r => And (IsNilpotent (HSub.hSub r x)) (Eq ((Polynomial.aev …
  -/
  refine existsUnique_of_exists_of_unique ?_ fun r₁ r₂ ⟨hr₁, hr₁'⟩ ⟨hr₂, hr₂'⟩ ↦ ?_
  · -- Existence
    /-
      case refine_1
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Polynomial R
      x : S
      h : IsNilpotent ((Polynomial.aeval x) P)
      h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
      ⊢ Exists fun x_1 => And (IsNilpotent (HSub.hSub x_1 x)) (Eq ((Polynomial.aeval …
    -/
    obtain ⟨n, hn⟩ := id h
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Polynomial R
      x : S
      h : IsNilpotent ((Polynomial.aeval x) P)
      h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
      n : Nat
      hn : Eq (HPow.hPow ((Polynomial.aeval x) P) n) 0
      ⊢ Exists fun x_1 => And (IsNilpotent (HSub.hSub x_1 x)) (Eq ((Polynomial.aeval …
    -/
    refine ⟨P.newtonMap^[n] x, isNilpotent_iterate_newtonMap_sub_of_isNilpotent h n, ?_⟩
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Polynomial R
      x : S
      h : IsNilpotent ((Polynomial.aeval x) P)
      h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
      n : Nat
      hn : Eq (HPow.hPow ((Polynomial.aeval x) P) n) 0
      ⊢ Eq ((Polynomial.aeval (Nat.iterate P.newtonMap n x)) P) 0
    -/
    rw [← zero_dvd_iff, ← pow_eq_zero_of_le (n.lt_two_pow_self).le hn]
    /-
      case refine_1.intro
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Polynomial R
      x : S
      h : IsNilpotent ((Polynomial.aeval x) P)
      h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
      n : Nat
      hn : Eq (HPow.hPow ((Polynomial.aeval x) P) n) 0
      ⊢ Dvd.dvd (HPow.hPow ((Polynomial.aeval x) P) (HPow.hPow 2 n)) ((Polynomial.ae …
    -/
    exact aeval_pow_two_pow_dvd_aeval_iterate_newtonMap h h' n
    /-
      🎉 no goals
    -/
  · -- Uniqueness
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Polynomial R
      x : S
      h : IsNilpotent ((Polynomial.aeval x) P)
      h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
      r₁ r₂ : S
      x✝¹ : And (IsNilpotent (HSub.hSub r₁ x)) (Eq ((Polynomial.aeval r₁) P) 0)
      x✝ : And (IsNilpotent (HSub.hSub r₂ x)) (Eq ((Polynomial.aeval r₂) P) 0)
      hr₁ : IsNilpotent (HSub.hSub r₁ x)
      hr₁' : Eq ((Polynomial.aeval r₁) P) 0
      hr₂ : IsNilpotent (HSub.hSub r₂ x)
      hr₂' : Eq ((Polynomial.aeval r₂) P) 0
      ⊢ Eq r₁ r₂
    -/
    have ⟨u, hu⟩ := binomExpansion (P.map (algebraMap R S)) r₁ (r₂ - r₁)
    suffices IsUnit (aeval r₁ (derivative P) + u * (r₂ - r₁)) by
      rwa [derivative_map, eval_map_algebraMap, eval_map_algebraMap, eval_map_algebraMap,
        add_sub_cancel, hr₂', hr₁', zero_add, pow_two, ← mul_assoc, ← add_mul, eq_comm,
        this.mul_right_eq_zero, sub_eq_zero, eq_comm] at hu
    have : IsUnit (aeval r₁ (derivative P)) :=
      isUnit_aeval_of_isUnit_aeval_of_isNilpotent_sub h' hr₁
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Polynomial R
      x : S
      h : IsNilpotent ((Polynomial.aeval x) P)
      h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
      r₁ r₂ : S
      x✝¹ : And (IsNilpotent (HSub.hSub r₁ x)) (Eq ((Polynomial.aeval r₁) P) 0)
      x✝ : And (IsNilpotent (HSub.hSub r₂ x)) (Eq ((Polynomial.aeval r₂) P) 0)
      hr₁ : IsNilpotent (HSub.hSub r₁ x)
      hr₁' : Eq ((Polynomial.aeval r₁) P) 0
      hr₂ : IsNilpotent (HSub.hSub r₂ x)
      hr₂' : Eq ((Polynomial.aeval r₂) P) 0
      u : S
      hu : Eq (Polynomial.eval (HAdd.hAdd r₁ (HSub.hSub r₂ r₁)) (Polynomial.map (alg …
      this : IsUnit ((Polynomial.aeval r₁) (Polynomial.derivative P))
      ⊢ IsUnit (HAdd.hAdd ((Polynomial.aeval r₁) (Polynomial.derivative P)) (HMul.hM …
    -/
    rw [← sub_sub_sub_cancel_right r₂ r₁ x]
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Polynomial R
      x : S
      h : IsNilpotent ((Polynomial.aeval x) P)
      h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
      r₁ r₂ : S
      x✝¹ : And (IsNilpotent (HSub.hSub r₁ x)) (Eq ((Polynomial.aeval r₁) P) 0)
      x✝ : And (IsNilpotent (HSub.hSub r₂ x)) (Eq ((Polynomial.aeval r₂) P) 0)
      hr₁ : IsNilpotent (HSub.hSub r₁ x)
      hr₁' : Eq ((Polynomial.aeval r₁) P) 0
      hr₂ : IsNilpotent (HSub.hSub r₂ x)
      hr₂' : Eq ((Polynomial.aeval r₂) P) 0
      u : S
      hu : Eq (Polynomial.eval (HAdd.hAdd r₁ (HSub.hSub r₂ r₁)) (Polynomial.map (alg …
      this : IsUnit ((Polynomial.aeval r₁) (Polynomial.derivative P))
      ⊢ IsUnit (HAdd.hAdd ((Polynomial.aeval r₁) (Polynomial.derivative P)) (HMul.hM …
    -/
    refine IsNilpotent.isUnit_add_left_of_commute ?_ this (Commute.all _ _)
    /-
      case refine_2
      R : Type u_1
      S : Type u_2
      inst✝² : CommRing R
      inst✝¹ : CommRing S
      inst✝ : Algebra R S
      P : Polynomial R
      x : S
      h : IsNilpotent ((Polynomial.aeval x) P)
      h' : IsUnit ((Polynomial.aeval x) (Polynomial.derivative P))
      r₁ r₂ : S
      x✝¹ : And (IsNilpotent (HSub.hSub r₁ x)) (Eq ((Polynomial.aeval r₁) P) 0)
      x✝ : And (IsNilpotent (HSub.hSub r₂ x)) (Eq ((Polynomial.aeval r₂) P) 0)
      hr₁ : IsNilpotent (HSub.hSub r₁ x)
      hr₁' : Eq ((Polynomial.aeval r₁) P) 0
      hr₂ : IsNilpotent (HSub.hSub r₂ x)
      hr₂' : Eq ((Polynomial.aeval r₂) P) 0
      u : S
      hu : Eq (Polynomial.eval (HAdd.hAdd r₁ (HSub.hSub r₂ r₁)) (Polynomial.map (alg …
      this : IsUnit ((Polynomial.aeval r₁) (Polynomial.derivative P))
      ⊢ IsNilpotent (HMul.hMul u (HSub.hSub (HSub.hSub r₂ x) (HSub.hSub r₁ x)))
    -/
    exact (Commute.all _ _).isNilpotent_mul_right <| (Commute.all _ _).isNilpotent_sub hr₂ hr₁
    /-
      🎉 no goals
    -/


@[deprecated (since := "2024-12-17")]
alias exists_unique_nilpotent_sub_and_aeval_eq_zero := existsUnique_nilpotent_sub_and_aeval_eq_zero


