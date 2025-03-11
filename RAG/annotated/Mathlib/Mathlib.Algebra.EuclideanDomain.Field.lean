instance (priority := 100) Field.toEuclideanDomain {K : Type*} [Field K] : EuclideanDomain K :=
{ toCommRing := Field.toCommRing
  quotient := (· / ·), remainder := fun a b => a - a * b / b, quotient_zero := div_zero,
  quotient_mul_add_remainder_eq := fun a b => by
    /-
      K : Type u_1
      inst✝ : Field K
      a b : K
      ⊢ Eq (HAdd.hAdd (HMul.hMul b ((fun x1 x2 => HDiv.hDiv x1 x2) a b)) ((fun a b = …
    -/
                           /-
                             🎉 no goals
                           -/
    by_cases h : b = 0 <;> simp [h, mul_div_cancel₀]
                           /-
                             🎉 no goals
                           -/
  r := fun a b => a = 0 ∧ b ≠ 0,
  r_wellFounded :=
    WellFounded.intro fun _ =>
      (Acc.intro _) fun _ ⟨hb, _⟩ => (Acc.intro _) fun _ ⟨_, hnb⟩ => False.elim <| hnb hb,
                                    /-
                                      K : Type u_1
                                      inst✝ : Field K
                                      a b : K
                                      hnb : Ne b 0
                                      ⊢ (fun a b => And (Eq a 0) (Ne b 0)) ((fun a b => HSub.hSub a (HDiv.hDiv (HMul …
                                    -/
  remainder_lt := fun a b hnb => by simp [hnb],
                                    /-
                                      🎉 no goals
                                    -/
  mul_left_not_lt := fun _ _ hnb ⟨hab, hna⟩ => Or.casesOn (mul_eq_zero.1 hab) hna hnb }

