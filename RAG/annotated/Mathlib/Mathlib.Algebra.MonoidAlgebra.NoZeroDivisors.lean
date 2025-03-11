/-- The coefficient of a monomial in a product `f * g` that can be reached in at most one way
as a product of monomials in the supports of `f` and `g` is a product. -/
theorem mul_apply_mul_eq_mul_of_uniqueMul [Mul A] {f g : MonoidAlgebra R A} {a0 b0 : A}
    (h : UniqueMul f.support g.support a0 b0) :
    (f * g) (a0 * b0) = f a0 * g b0 := by
  classical
  simp_rw [mul_apply, sum, ← Finset.sum_product']
  refine (Finset.sum_eq_single (a0, b0) ?_ ?_).trans (if_pos rfl) <;> simp_rw [Finset.mem_product]
  · refine fun ab hab hne => if_neg (fun he => hne <| Prod.ext ?_ ?_)
    exacts [(h hab.1 hab.2 he).1, (h hab.1 hab.2 he).2]
  · refine fun hnmem => ite_eq_right_iff.mpr (fun _ => ?_)
    rcases not_and_or.mp hnmem with af | bg
    · rw [not_mem_support_iff.mp af, zero_mul]
    · rw [not_mem_support_iff.mp bg, mul_zero]


instance instNoZeroDivisorsOfUniqueProds [NoZeroDivisors R] [Mul A] [UniqueProds A] :
    NoZeroDivisors (MonoidAlgebra R A) where
  eq_zero_or_eq_zero_of_mul_eq_zero {a b} ab := by
    /-
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : NoZeroDivisors R
      inst✝¹ : Mul A
      inst✝ : UniqueProds A
      a b : MonoidAlgebra R A
      ab : Eq (HMul.hMul a b) 0
      ⊢ Or (Eq a 0) (Eq b 0)
    -/
    contrapose! ab
    obtain ⟨da, a0, db, b0, h⟩ := UniqueProds.uniqueMul_of_nonempty
      (support_nonempty_iff.mpr ab.1) (support_nonempty_iff.mpr ab.2)
    /-
      case intro.intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : NoZeroDivisors R
      inst✝¹ : Mul A
      inst✝ : UniqueProds A
      a b : MonoidAlgebra R A
      ab : And (Ne a 0) (Ne b 0)
      da : A
      a0 : Membership.mem a.support da
      db : A
      b0 : Membership.mem b.support db
      h : UniqueMul a.support b.support da db
      ⊢ Ne (HMul.hMul a b) 0
    -/
    refine support_nonempty_iff.mp ⟨da * db, ?_⟩
    /-
      case intro.intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : NoZeroDivisors R
      inst✝¹ : Mul A
      inst✝ : UniqueProds A
      a b : MonoidAlgebra R A
      ab : And (Ne a 0) (Ne b 0)
      da : A
      a0 : Membership.mem a.support da
      db : A
      b0 : Membership.mem b.support db
      h : UniqueMul a.support b.support da db
      ⊢ Membership.mem (HMul.hMul a b).support (HMul.hMul da db)
    -/
    rw [mem_support_iff] at a0 b0 ⊢
    /-
      case intro.intro.intro.intro
      R : Type u_1
      A : Type u_2
      inst✝³ : Semiring R
      inst✝² : NoZeroDivisors R
      inst✝¹ : Mul A
      inst✝ : UniqueProds A
      a b : MonoidAlgebra R A
      ab : And (Ne a 0) (Ne b 0)
      da : A
      a0 : Ne (a da) 0
      db : A
      b0 : Ne (b db) 0
      h : UniqueMul a.support b.support da db
      ⊢ Ne ((HMul.hMul a b) (HMul.hMul da db)) 0
    -/
    exact mul_apply_mul_eq_mul_of_uniqueMul h ▸ mul_ne_zero a0 b0
    /-
      🎉 no goals
    -/


/-- The coefficient of a monomial in a product `f * g` that can be reached in at most one way
as a product of monomials in the supports of `f` and `g` is a product. -/
theorem mul_apply_add_eq_mul_of_uniqueAdd [Add A] {f g : R[A]} {a0 b0 : A}
    (h : UniqueAdd f.support g.support a0 b0) :
    (f * g) (a0 + b0) = f a0 * g b0 :=
  MonoidAlgebra.mul_apply_mul_eq_mul_of_uniqueMul (A := Multiplicative A) h


instance [NoZeroDivisors R] [Add A] [UniqueSums A] : NoZeroDivisors R[A] :=
  MonoidAlgebra.instNoZeroDivisorsOfUniqueProds (A := Multiplicative A)


